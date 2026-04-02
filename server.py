import asyncio
import uuid
from datetime import datetime
from typing import Optional
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

LM_STUDIO_URL = "http://10.0.0.1:1234/v1/chat/completions"
MODEL_ID = "qwen2.5-14b-instruct"
SYSTEM_PROMPT = "You are a helpful local assistant running on a Raspberry Pi. Be concise and practical."

app = FastAPI()

# In-memory job queue
queue: deque = deque()
jobs: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class Job(BaseModel):
    id: str
    message: str
    status: str  # queued | processing | done | error
    result: Optional[str] = None
    created_at: str


async def call_lm_studio(messages: list, stream: bool = False):
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "stream": stream,
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


# Session memory (simple in-memory, keyed by session_id)
sessions: dict = {}


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    sessions[session_id].append({"role": "user", "content": req.message})

    if req.stream:
        async def stream_response():
            full = ""
            async for chunk in call_lm_studio(sessions[session_id], stream=True):
                import json
                try:
                    data = json.loads(chunk)
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
        async for chunk in call_lm_studio(sessions[session_id]):
            result = chunk
        sessions[session_id].append({"role": "assistant", "content": result})
        return {"reply": result, "session_id": session_id}


@app.post("/queue")
async def enqueue(req: ChatRequest):
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id": job_id,
        "message": req.message,
        "status": "queued",
        "result": None,
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


@app.get("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "queued_jobs": len(queue)}


# Background queue worker
async def queue_worker():
    while True:
        if queue:
            job_id = queue.popleft()
            job = jobs[job_id]
            job["status"] = "processing"
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": job["message"]},
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
