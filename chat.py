#!/usr/bin/env python3
"""CLI chat client — streams responses from the local assistant server."""

import sys
import httpx

SERVER = "http://localhost:8000"
SESSION = "cli"


def chat(message: str):
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{SERVER}/chat",
            json={"message": message, "session_id": SESSION, "stream": True},
        ) as r:
            for chunk in r.iter_text():
                print(chunk, end="", flush=True)
    print()


def main():
    if len(sys.argv) > 1:
        # One-shot: python chat.py "your prompt"
        chat(" ".join(sys.argv[1:]))
        return

    print("Local assistant ready. Ctrl+C to exit.\n")
    while True:
        try:
            msg = input("you: ").strip()
            if not msg:
                continue
            print("ai: ", end="", flush=True)
            chat(msg)
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
