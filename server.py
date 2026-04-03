import asyncio
import uuid
import tempfile
import pty
import os
import fcntl
import termios
import struct
import json
from datetime import datetime
from typing import Optional
from collections import deque
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PIPER_BIN        = Path(__file__).parent / "piper/piper/piper"
PIPER_VOICES_DIR = Path(__file__).parent / "piper/voices"

LM_STUDIO_URL  = "http://10.0.0.1:1234/v1/chat/completions"
MODEL_ID       = "qwen2.5-14b-instruct"
SYSTEM_PROMPT  = "You are a helpful local assistant running on a Raspberry Pi. Be concise and practical."
HISTORY_LIMIT  = 10  # number of user+assistant turns to keep per session
CLAUDE_CLI     = "claude"

# ── TTS config (mutable at runtime) ──────────────────────────────────────────
tts_config = {
    "voice":           "en_US-lessac-medium",
    "length_scale":    1.25,
    "initial_words":   15,
    "subsequent_words": 3,
}

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
async def index():
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ── Job queue ─────────────────────────────────────────────────────────────────
queue: deque = deque()
jobs:  dict  = {}

# ── Session memory ────────────────────────────────────────────────────────────
sessions: dict = {}


# ── Models ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None
    stream:     bool = False


class TTSRequest(BaseModel):
    text: str


class TTSConfig(BaseModel):
    voice:            Optional[str]   = None
    length_scale:     Optional[float] = None
    initial_words:    Optional[int]   = None
    subsequent_words: Optional[int]   = None


# ── LM Studio ────────────────────────────────────────────────────────────────
async def call_lm_studio(messages: list, stream: bool = False):
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model":      MODEL_ID,
            "messages":   messages,
            "stream":     stream,
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        if stream:
            async with client.stream("POST", LM_STUDIO_URL, json=payload) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        yield line[6:]
        else:
            r = await client.post(LM_STUDIO_URL, json=payload)
            r.raise_for_status()
            yield r.json()["choices"][0]["message"]["content"]


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    sessions[session_id].append({"role": "user", "content": req.message})

    # Keep system prompt + last HISTORY_LIMIT turns (2 messages per turn)
    system = sessions[session_id][:1]
    history = sessions[session_id][1:]
    if len(history) > HISTORY_LIMIT * 2:
        history = history[-(HISTORY_LIMIT * 2):]
    trimmed = system + history

    if req.stream:
        async def stream_response():
            import json
            full = ""
            async for chunk in call_lm_studio(trimmed, stream=True):
                try:
                    data  = json.loads(chunk)
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        full += token
                        yield token
                except Exception:
                    pass
            sessions[session_id].append({"role": "assistant", "content": full})

        return StreamingResponse(stream_response(), media_type="text/plain")
    else:
        result = ""
        async for chunk in call_lm_studio(trimmed):
            result = chunk
        sessions[session_id].append({"role": "assistant", "content": result})
        return {"reply": result, "session_id": session_id}


# ── Queue ─────────────────────────────────────────────────────────────────────
@app.post("/queue")
async def enqueue(req: ChatRequest):
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id":         job_id,
        "message":    req.message,
        "status":     "queued",
        "result":     None,
        "created_at": datetime.now().isoformat(),
    }
    jobs[job_id] = job
    queue.append(job_id)
    return {"job_id": job_id}


@app.get("/queue/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ── Sessions ──────────────────────────────────────────────────────────────────
@app.get("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}


# ── TTS ───────────────────────────────────────────────────────────────────────
@app.get("/tts/voices")
async def list_voices():
    voices = sorted(
        p.stem for p in PIPER_VOICES_DIR.glob("*.onnx")
        if p.suffix == ".onnx" and not p.name.endswith(".onnx.json")
    )
    return {"voices": voices, "current": tts_config["voice"]}


@app.get("/tts/config")
async def get_tts_config():
    return tts_config


@app.post("/tts/config")
async def set_tts_config(cfg: TTSConfig):
    if cfg.voice is not None:
        model_path = PIPER_VOICES_DIR / f"{cfg.voice}.onnx"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice not found: {cfg.voice}")
        tts_config["voice"] = cfg.voice
    if cfg.length_scale is not None:
        tts_config["length_scale"] = max(0.5, min(2.0, cfg.length_scale))
    if cfg.initial_words is not None:
        tts_config["initial_words"] = max(1, cfg.initial_words)
    if cfg.subsequent_words is not None:
        tts_config["subsequent_words"] = max(1, cfg.subsequent_words)
    return tts_config


@app.post("/tts")
async def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    model_path = PIPER_VOICES_DIR / f"{tts_config['voice']}.onnx"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    proc = await asyncio.create_subprocess_exec(
        str(PIPER_BIN),
        "--model",        str(model_path),
        "--length-scale", str(tts_config["length_scale"]),
        "--output_file",  out_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate(input=text.encode())

    audio = Path(out_path).read_bytes()
    Path(out_path).unlink(missing_ok=True)
    return Response(content=audio, media_type="audio/wav")


# ── Health ────────────────────────────────────────────────────────────────────
@app.post("/claude")
async def claude_chat(req: ChatRequest):
    session_id = req.session_id or "claude-default"
    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append({"role": "user", "content": req.message})

    # Build conversation context as a single prompt string for claude -p
    history = sessions[session_id][-(HISTORY_LIMIT * 2):]
    context = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[:-1]
    )
    prompt = (context + "\n\nUser: " + req.message).strip() if context else req.message

    if req.stream:
        async def stream_claude():
            proc = await asyncio.create_subprocess_exec(
                CLAUDE_CLI, "-p", prompt,
                "--dangerously-skip-permissions",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            full = ""
            async for line in proc.stdout:
                token = line.decode()
                full += token
                yield token
            await proc.wait()
            sessions[session_id].append({"role": "assistant", "content": full.strip()})

        return StreamingResponse(stream_claude(), media_type="text/plain")
    else:
        proc = await asyncio.create_subprocess_exec(
            CLAUDE_CLI, "-p", prompt,
            "--dangerously-skip-permissions",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        result = stdout.decode().strip()
        sessions[session_id].append({"role": "assistant", "content": result})
        return {"reply": result, "session_id": session_id}


@app.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    master_fd, slave_fd = pty.openpty()

    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["HOME"] = os.path.expanduser("~")

    process = await asyncio.create_subprocess_exec(
        "bash", "--login",
        stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
        close_fds=True, env=env,
        cwd=os.path.expanduser("~"),
    )
    os.close(slave_fd)

    def set_winsize(fd, rows, cols):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    set_winsize(master_fd, 24, 80)

    async def read_pty():
        while True:
            try:
                data = await loop.run_in_executor(None, lambda: os.read(master_fd, 4096))
                await websocket.send_bytes(data)
            except Exception:
                break

    async def write_pty():
        while True:
            try:
                msg = await websocket.receive()
                if "text" in msg:
                    try:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "resize":
                            set_winsize(master_fd, ctrl["rows"], ctrl["cols"])
                    except Exception:
                        pass
                elif "bytes" in msg:
                    os.write(master_fd, msg["bytes"])
            except WebSocketDisconnect:
                break
            except Exception:
                break

    read_task  = asyncio.create_task(read_pty())
    write_task = asyncio.create_task(write_pty())
    done, pending = await asyncio.wait([read_task, write_task], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()

    try:
        process.terminate()
    except Exception:
        pass
    try:
        os.close(master_fd)
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "queued_jobs": len(queue)}


# ── Background queue worker ───────────────────────────────────────────────────
async def queue_worker():
    while True:
        if queue:
            job_id = queue.popleft()
            job    = jobs[job_id]
            job["status"] = "processing"
            try:
                messages = [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": job["message"]},
                ]
                result = ""
                async for chunk in call_lm_studio(messages):
                    result = chunk
                job["result"] = result
                job["status"] = "done"
            except Exception as e:
                job["status"] = "error"
                job["result"] = str(e)
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(queue_worker())
