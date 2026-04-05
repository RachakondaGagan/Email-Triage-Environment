"""
Email Triage Environment — Python Client
=========================================
Drop-in client for OpenEnv-compatible training code.
Supports both async and sync usage.

Usage (async):
    async with EmailTriageEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset(task_name="triage_and_reply")
        result = await env.step(EmailTriageAction(...))

Usage (sync, for notebooks/scripts):
    with EmailTriageEnv(base_url="...").sync() as env:
        result = env.reset()
        result = env.step(EmailTriageAction(...))

Local Docker usage:
    env = await EmailTriageEnv.from_docker_image("email-triage-env")
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import websockets

from models import (
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    StepResult,
)


class EmailTriageEnv:
    """Async WebSocket client for the Email Triage environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self._base_url = base_url.rstrip("/")
        self._ws_url = self._base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        self._ws = None

    # ------------------------------------------------------------------ #
    #  Context manager
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "EmailTriageEnv":
        self._ws = await websockets.connect(self._ws_url)
        return self

    async def __aexit__(self, *args):
        if self._ws:
            await self._ws.close()

    # ------------------------------------------------------------------ #
    #  Core API
    # ------------------------------------------------------------------ #

    async def _send(self, method: str, params: Dict[str, Any]) -> Dict:
        payload = json.dumps({"method": method, "params": params})
        await self._ws.send(payload)
        raw = await self._ws.recv()
        response = json.loads(raw)
        if response.get("error"):
            raise RuntimeError(f"Server error: {response['error']}")
        return response["result"]

    async def reset(self, task_name: str = "triage_and_reply") -> StepResult:
        result = await self._send("reset", {"task_name": task_name})
        obs = EmailTriageObservation(**result["observation"])
        return StepResult(
            observation=obs,
            reward=result.get("reward", 0.0),
            done=result.get("done", False),
            info=result.get("info", {}),
        )

    async def step(self, action: EmailTriageAction) -> StepResult:
        result = await self._send("step", {"action": action.model_dump()})
        obs = EmailTriageObservation(**result["observation"])
        return StepResult(
            observation=obs,
            reward=result.get("reward", 0.0),
            done=result.get("done", False),
            info=result.get("info", {}),
        )

    async def state(self) -> EmailTriageState:
        result = await self._send("state", {})
        return EmailTriageState(**result)

    async def close(self):
        if self._ws:
            await self._ws.close()

    # ------------------------------------------------------------------ #
    #  Sync wrapper
    # ------------------------------------------------------------------ #

    def sync(self) -> "_SyncEmailTriageEnv":
        return _SyncEmailTriageEnv(self._base_url)

    # ------------------------------------------------------------------ #
    #  Docker factory
    # ------------------------------------------------------------------ #

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "EmailTriageEnv":
        """Start a local Docker container and connect to it."""
        import subprocess
        import time

        container_name = f"email-triage-env-{port}"
        subprocess.run(
            ["docker", "run", "-d", "--name", container_name,
             "-p", f"{port}:7860", image_name],
            check=True, capture_output=True,
        )
        # Wait for server to start
        time.sleep(3)
        env = cls(base_url=f"http://localhost:{port}")
        env._ws = await websockets.connect(env._ws_url)
        return env


class _SyncEmailTriageEnv:
    """Synchronous wrapper around the async client."""

    def __init__(self, base_url: str):
        self._base_url = base_url
        self._async_env: Optional[EmailTriageEnv] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._async_env = self._loop.run_until_complete(
            EmailTriageEnv(self._base_url).__aenter__()
        )
        return self

    def __exit__(self, *args):
        if self._async_env and self._loop:
            self._loop.run_until_complete(self._async_env.__aexit__(*args))
            self._loop.close()

    def reset(self, task_name: str = "triage_and_reply") -> StepResult:
        return self._loop.run_until_complete(self._async_env.reset(task_name=task_name))

    def step(self, action: EmailTriageAction) -> StepResult:
        return self._loop.run_until_complete(self._async_env.step(action))

    def state(self) -> EmailTriageState:
        return self._loop.run_until_complete(self._async_env.state())