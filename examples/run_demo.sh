#!/usr/bin/env bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Then POST the file:
# curl -X POST "http://localhost:8000/process" -F "file=@examples/sample_conversation.mp3"
