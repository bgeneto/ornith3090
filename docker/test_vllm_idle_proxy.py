#!/usr/bin/env python3
"""Fake-upstream tests for docker/vllm-idle-proxy.py. No GPU.

  venv/bin/python docker/test_vllm_idle_proxy.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "vllm_idle_proxy", ROOT / "vllm-idle-proxy.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["vllm_idle_proxy"] = mod
spec.loader.exec_module(mod)
Config = mod.Config
IdleProxy = mod.IdleProxy


class FakeEngine:
    def __init__(self, *, sleeping: bool = False, health_when_sleeping: int = 503):
        self.sleeping = sleeping
        self.health_when_sleeping = health_when_sleeping
        self.wake_calls = 0
        self.sleep_calls: list[tuple[str, str]] = []
        self.lock = asyncio.Lock()

    async def health(self, request: Request) -> Response:
        if self.sleeping:
            return Response(b"sleeping", status_code=self.health_when_sleeping)
        return Response(b"ok", status_code=200)

    async def is_sleeping(self, request: Request) -> JSONResponse:
        return JSONResponse({"is_sleeping": self.sleeping})

    async def wake_up(self, request: Request) -> JSONResponse:
        async with self.lock:
            self.wake_calls += 1
            await asyncio.sleep(0.15)
            self.sleeping = False
        return JSONResponse({"ok": True})

    async def sleep(self, request: Request) -> JSONResponse:
        self.sleep_calls.append(
            (request.query_params.get("level", ""), request.query_params.get("mode", ""))
        )
        self.sleeping = True
        return JSONResponse({"ok": True})

    async def models(self, request: Request) -> JSONResponse:
        return JSONResponse({"data": [{"id": "ornith-1.5-9b"}]})

    async def chat(self, request: Request) -> Response:
        if self.sleeping:
            return JSONResponse(
                {
                    "error": {
                        "message": "Model is currently in sleep mode. Please wake it up first",
                        "type": "ModelSleepingError",
                    }
                },
                status_code=503,
            )

        async def chunks():
            yield b'data: {"id":"x","choices":[{"delta":{"content":"OK"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    def app(self) -> Starlette:
        methods = ["GET", "POST"]
        return Starlette(
            routes=[
                Route("/health", self.health, methods=methods),
                Route("/is_sleeping", self.is_sleeping, methods=methods),
                Route("/wake_up", self.wake_up, methods=methods),
                Route("/sleep", self.sleep, methods=methods),
                Route("/v1/models", self.models, methods=["GET"]),
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
            ]
        )


class UvicornThread(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=0):
        super().__init__(daemon=True)
        self.config = uvicorn.Config(
            app, host=host, port=port, log_level="error", lifespan="on"
        )
        self.server = uvicorn.Server(self.config)
        self._port = port

    def run(self):
        self.server.run()

    def wait_started(self, timeout=8.0):
        deadline = time.time() + timeout
        while not self.server.started:
            if time.time() > deadline:
                raise TimeoutError("uvicorn did not start")
            time.sleep(0.02)
        sock = self.server.servers[0].sockets[0]
        self._port = sock.getsockname()[1]

    @property
    def port(self) -> int:
        return self._port

    def stop(self):
        self.server.should_exit = True


class ProxyStack:
    def __init__(self, engine: FakeEngine, **cfg):
        self.engine = engine
        self.engine_http = UvicornThread(engine.app())
        self.engine_http.start()
        self.engine_http.wait_started()
        defaults = dict(
            upstream=f"http://127.0.0.1:{self.engine_http.port}",
            bind_host="127.0.0.1",
            bind_port=0,
            idle_timeout=60.0,
            poll_seconds=0.1,
            sleep_level=1,
            connect_timeout=2.0,
            wake_http_timeout=5.0,
            sleep_http_timeout=5.0,
        )
        defaults.update(cfg)
        self.proxy = IdleProxy(Config(**defaults))
        self.proxy_http = UvicornThread(self.proxy.app)
        self.proxy_http.start()
        self.proxy_http.wait_started()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_http.port}"

    def close(self):
        self.proxy_http.stop()
        self.proxy_http.join(timeout=8)
        self.engine_http.stop()
        self.engine_http.join(timeout=8)


class IdleProxyTests(unittest.TestCase):
    def tearDown(self):
        stack = getattr(self, "stack", None)
        if stack is not None:
            stack.close()

    def test_exempt_models_does_not_wake_or_reset_idle(self):
        self.stack = ProxyStack(FakeEngine(sleeping=True))
        r = httpx.get(self.stack.url + "/v1/models", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["data"][0]["id"], "ornith-1.5-9b")
        started = self.stack.proxy.last_activity
        r = httpx.get(self.stack.url + "/v1/models", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.stack.engine.wake_calls, 0)
        self.assertEqual(self.stack.proxy.last_activity, started)
        self.assertTrue(self.stack.engine.sleeping)

    def test_exempt_health_does_not_wake(self):
        self.stack = ProxyStack(FakeEngine(sleeping=True))
        r = httpx.get(self.stack.url + "/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.stack.engine.wake_calls, 0)

    def test_chat_wakes_once_under_lock_then_streams(self):
        self.stack = ProxyStack(FakeEngine(sleeping=True))

        def chat():
            return httpx.post(
                self.stack.url + "/v1/chat/completions",
                json={"model": "ornith-1.5-9b", "messages": []},
                timeout=10,
            )

        results = []

        def run():
            results.append(chat())

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.status_code, 200)
            self.assertIn("OK", r.text)
        self.assertEqual(self.stack.engine.wake_calls, 1)
        self.assertFalse(self.stack.engine.sleeping)

    def test_idle_sleep_when_quiet(self):
        self.stack = ProxyStack(
            FakeEngine(sleeping=False), idle_timeout=0.35, poll_seconds=0.08
        )
        r = httpx.get(self.stack.url + "/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(self.stack.proxy.engine_ready)
        deadline = time.time() + 3.0
        while time.time() < deadline and not self.stack.engine.sleep_calls:
            time.sleep(0.05)
        self.assertEqual(self.stack.engine.sleep_calls, [("1", "wait")])
        self.assertTrue(self.stack.engine.sleeping)

    def test_public_sleep_and_rpc_are_blocked(self):
        self.stack = ProxyStack(FakeEngine(sleeping=False))
        for path in ("/sleep", "/wake_up", "/collective_rpc", "/pause", "/resume"):
            r = httpx.post(self.stack.url + path, timeout=5)
            self.assertEqual(r.status_code, 404, path)
        self.assertEqual(self.stack.engine.sleep_calls, [])
        self.assertEqual(self.stack.engine.wake_calls, 0)

    def test_health_200_while_engine_sleeping_and_upstream_503(self):
        self.stack = ProxyStack(FakeEngine(sleeping=True, health_when_sleeping=503))
        r = httpx.get(self.stack.url + "/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.stack.engine.wake_calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
