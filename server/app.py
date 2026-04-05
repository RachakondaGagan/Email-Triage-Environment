"""
Email Triage Environment — FastAPI Server
==========================================
Exposes:
  GET  /health              → liveness check
  GET  /info                → environment metadata
  POST /reset               → start a new episode
  POST /step                → take one action
  GET  /state               → current episode state
  WS   /ws                  → WebSocket session (reset / step / state)
"""
from __future__ import annotations

import json
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    StepResult,
)
from server.environment import TASKS, EmailTriageEnvironment


# ---------------------------------------------------------------------------
# Session store (in-memory)
# ---------------------------------------------------------------------------

_sessions: Dict[str, EmailTriageEnvironment] = {}


def _get_or_create_session(session_id: str, task_name: str = "triage_and_reply") -> EmailTriageEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = EmailTriageEnvironment(task_name=task_name)
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _sessions.clear()


app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "A real-world email support triage environment for RL agent training. "
        "Agents classify emails, draft replies, escalate incidents, and handle compliance. "
        "Three tasks: classify_only (easy), triage_and_reply (medium), crisis_response (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "email_triage", "version": "1.0.0"}


@app.get("/info")
async def info():
    return {
        "name": "email_triage_env",
        "description": "Email support triage environment for RL agent evaluation",
        "tasks": {
            name: {
                "description": cfg["description"][:120] + "...",
                "num_emails": len(cfg["email_ids"]),
                "max_steps": cfg["max_steps"],
                "difficulty": cfg["difficulty"],
            }
            for name, cfg in TASKS.items()
        },
        "action_types": ["classify", "draft_reply", "escalate", "archive", "request_info"],
        "categories": ["billing", "technical", "security", "general_inquiry", "spam",
                        "outage", "feature_request", "compliance"],
        "priorities": ["critical", "high", "medium", "low"],
        "spec_version": "1",
    }


class ResetRequest(BaseModel):
    task_name: Optional[str] = "triage_and_reply"
    session_id: Optional[str] = None


@app.post("/reset")
async def reset(req: ResetRequest):
    session_id = req.session_id or str(uuid.uuid4())
    env = _get_or_create_session(session_id, req.task_name or "triage_and_reply")
    try:
        obs = env.reset(task_name=req.task_name)
        return {
            "session_id": session_id,
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {"task": req.task_name},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str


@app.post("/step")
async def step(req: StepRequest):
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found. Call /reset first.")
    env = _sessions[req.session_id]
    try:
        action = EmailTriageAction(**req.action)
        result = env.step(action)
        resp = result.model_dump()
        if result.done:
            resp["final_score"] = env.final_score()
        return resp
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
async def state(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    env = _sessions[session_id]
    try:
        return env.state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket protocol:
      Client sends JSON: {"method": "reset"|"step"|"state", "params": {...}}
      Server responds:   {"result": {...}, "error": null}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = EmailTriageEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            method = msg.get("method")
            params = msg.get("params", {})
            result: Any = None
            error: Optional[str] = None

            try:
                if method == "reset":
                    task_name = params.get("task_name", "triage_and_reply")
                    obs = env.reset(task_name=task_name)
                    result = {
                        "session_id": session_id,
                        "observation": obs.model_dump(),
                        "reward": 0.0,
                        "done": False,
                    }

                elif method == "step":
                    action = EmailTriageAction(**params.get("action", {}))
                    step_result = env.step(action)
                    result = step_result.model_dump()
                    if step_result.done:
                        result["final_score"] = env.final_score()

                elif method == "state":
                    result = env.state.model_dump()

                else:
                    error = f"Unknown method '{method}'. Use reset/step/state."

            except (ValidationError, ValueError, RuntimeError) as e:
                error = str(e)
            except Exception as e:
                error = traceback.format_exc()

            await websocket.send_text(json.dumps({"result": result, "error": error}))

    except WebSocketDisconnect:
        pass
    finally:
        if session_id in _sessions:
            del _sessions[session_id]