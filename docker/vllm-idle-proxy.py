#!/usr/bin/env python3
"""Wake-on-request / idle-sleep reverse proxy in front of a vLLM engine.

vLLM's POST /sleep parks the GPU but does not wake on the next chat request.
This process owns both halves: it tracks real inference activity, POSTs /sleep
after VLLM_IDLE_TIMEOUT, and POST /wake_up before forwarding a non-exempt
request. Docker healthchecks keep hitting GET /health on the public port.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Iterable, Mapping

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp

log = logging.getLogger("vllm-idle-proxy")

EXEMPT_ANY_METHOD = frozenset(
    {
        "/health",
        "/ping",
        "/version",
        "/load",
        "/metrics",
        "/is_sleeping",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)
EXEMPT_GET = frozenset({"/v1/models"})
BLOCKED = frozenset(
    {"/sleep", "/wake_up", "/collective_rpc", "/pause", "/resume"}
)
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _norm_path(path: str) -> str:
    if not path:
        return "/"
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/")
    return path


def is_exempt(method: str, path: str) -> bool:
    path = _norm_path(path)
    if path in EXEMPT_ANY_METHOD:
        return True
    if method.upper() == "GET" and path in EXEMPT_GET:
        return True
    return False


def is_blocked(path: str) -> bool:
    path = _norm_path(path)
    if path in BLOCKED:
        return True
    for prefix in ("/collective_rpc",):
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _filter_req_headers(headers: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP:
            continue
        out[key] = value
    return out


def _filter_resp_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers:
        if key.lower() in HOP_BY_HOP:
            continue
        out[key] = value
    return out


def _sleeping_status(status_code: int, body: bytes | None = None) -> bool:
    if status_code != 503:
        return False
    if not body:
        return True
    text = body.decode("utf-8", errors="replace").lower()
    return "sleep" in text or "sleeping" in text


@dataclass
class Config:
    upstream: str
    bind_host: str = "0.0.0.0"
    bind_port: int = 8000
    idle_timeout: float = 90.0
    poll_seconds: float = 15.0
    sleep_level: int = 1
    api_key: str | None = None
    sleep_http_timeout: float = 300.0
    wake_http_timeout: float = 180.0
    connect_timeout: float = 5.0

    @classmethod
    def from_env(cls) -> "Config":
        bind_port = int(os.getenv("VLLM_IDLE_BIND_PORT", os.getenv("PORT", "8000")))
        engine_port = int(os.getenv("VLLM_ENGINE_PORT", str(bind_port + 1)))
        upstream = os.getenv(
            "VLLM_IDLE_UPSTREAM", f"http://127.0.0.1:{engine_port}"
        ).rstrip("/")
        api_key = os.getenv("VLLM_API_KEY") or None
        return cls(
            upstream=upstream,
            bind_host=os.getenv("VLLM_IDLE_BIND_HOST", os.getenv("HOST", "0.0.0.0")),
            bind_port=bind_port,
            idle_timeout=float(os.getenv("VLLM_IDLE_TIMEOUT", "90")),
            poll_seconds=float(os.getenv("VLLM_IDLE_POLL_SECONDS", "15")),
            sleep_level=int(os.getenv("SLEEP_LEVEL", os.getenv("VLLM_SLEEP_LEVEL", "1"))),
            api_key=api_key,
            sleep_http_timeout=float(os.getenv("VLLM_IDLE_SLEEP_TIMEOUT", "300")),
            wake_http_timeout=float(os.getenv("VLLM_IDLE_WAKE_TIMEOUT", "180")),
            connect_timeout=float(os.getenv("VLLM_IDLE_HTTP_TIMEOUT", "5")),
        )


class IdleProxy:
    def __init__(self, config: Config):
        self.config = config
        self.wake_lock = asyncio.Lock()
        self.inflight = 0
        self.last_activity = time.monotonic()
        self.engine_ready = False
        self._stopped = False
        self.client: httpx.AsyncClient | None = None
        self.app = self._build_app()

    def _auth_headers(self) -> dict[str, str]:
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {}

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("proxy client is not started")
        return self.client

    def mark_ready(self) -> None:
        if not self.engine_ready:
            self.engine_ready = True
            self.last_activity = time.monotonic()
            log.info("engine ready; idle timer starts (%ss)", self.config.idle_timeout)

    async def startup(self) -> None:
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=None,
            write=60.0,
            pool=5.0,
        )
        self.client = httpx.AsyncClient(
            base_url=self.config.upstream,
            timeout=timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=32),
            follow_redirects=False,
        )
        self.last_activity = time.monotonic()
        log.info(
            "idle proxy listening %s:%s -> %s (sleep level %s, timeout %ss)",
            self.config.bind_host,
            self.config.bind_port,
            self.config.upstream,
            self.config.sleep_level,
            self.config.idle_timeout,
        )

    async def shutdown(self) -> None:
        self._stopped = True
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def is_sleeping(self) -> bool:
        try:
            response = await self._client().get(
                "/is_sleeping",
                headers=self._auth_headers(),
                timeout=self.config.connect_timeout,
            )
            if response.status_code == 200:
                self.mark_ready()
                data = response.json()
                return bool(data.get("is_sleeping"))
        except Exception as exc:
            log.debug("is_sleeping failed: %s", exc)
        return False

    async def ensure_awake(self) -> None:
        async with self.wake_lock:
            if not await self.is_sleeping():
                return
            log.info("vLLM sleeping -> POST /wake_up")
            await self._client().post(
                "/wake_up",
                headers=self._auth_headers(),
                timeout=self.config.wake_http_timeout,
            )
            deadline = time.monotonic() + self.config.wake_http_timeout
            while time.monotonic() < deadline:
                if not await self.is_sleeping():
                    log.info("vLLM awake")
                    return
                await asyncio.sleep(0.1)
            raise RuntimeError(
                f"vLLM still sleeping after {self.config.wake_http_timeout:.0f}s"
            )

    async def idle_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.config.poll_seconds)
            if self._stopped:
                return
            try:
                await self._maybe_sleep()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._stopped:
                    log.warning("idle loop error: %s", exc)

    async def _maybe_sleep(self) -> None:
        if self._stopped or self.client is None:
            return
        if not self.engine_ready:
            return
        if self.inflight > 0:
            return
        idle = time.monotonic() - self.last_activity
        if idle < self.config.idle_timeout:
            return
        if await self.is_sleeping():
            return
        log.info(
            "vLLM idle for %.0fs -> sleep level %s",
            idle,
            self.config.sleep_level,
        )
        try:
            response = await self._client().post(
                "/sleep",
                params={"level": self.config.sleep_level, "mode": "wait"},
                headers=self._auth_headers(),
                timeout=self.config.sleep_http_timeout,
            )
            response.raise_for_status()
            log.info("vLLM sleeping")
        except Exception as exc:
            if self._stopped or self.client is None:
                return
            log.warning("Failed to put vLLM to sleep: %s", exc)

    async def handle_health(self, request: Request) -> Response:
        sleeping = await self.is_sleeping()
        try:
            response = await self._client().get(
                "/health",
                timeout=self.config.connect_timeout,
            )
            if response.status_code == 200:
                self.mark_ready()
                return Response(
                    content=response.content,
                    status_code=200,
                    headers=_filter_resp_headers(response.headers.items()),
                )
            if sleeping:
                return Response(status_code=200)
            return Response(content=response.content, status_code=response.status_code)
        except Exception as exc:
            if sleeping:
                log.debug("health synthesized 200 while sleeping: %s", exc)
                return Response(status_code=200)
            return Response(content=b"upstream unavailable\n", status_code=503)

    async def handle_blocked(self, request: Request) -> Response:
        return JSONResponse({"error": "not found"}, status_code=404)

    async def handle_proxy(self, request: Request) -> Response:
        if is_blocked(request.url.path):
            return await self.handle_blocked(request)
        if _norm_path(request.url.path) == "/health":
            return await self.handle_health(request)

        wake = not is_exempt(request.method, request.url.path)
        if wake:
            await self.ensure_awake()
            self.inflight += 1
            self.last_activity = time.monotonic()
        try:
            return await self._forward(request, retry=wake)
        finally:
            if wake:
                self.inflight = max(0, self.inflight - 1)
                self.last_activity = time.monotonic()

    async def _forward(self, request: Request, *, retry: bool) -> Response:
        body = await request.body()
        headers = _filter_req_headers(request.headers)
        url = request.url.path
        if request.url.query:
            url = f"{url}?{request.url.query}"

        cm = self._client().stream(
            request.method,
            url,
            headers=headers,
            content=body if body else None,
        )
        resp = await cm.__aenter__()
        try:
            if retry and _sleeping_status(resp.status_code):
                await resp.aread()
                await cm.__aexit__(None, None, None)
                cm = None
                await self.ensure_awake()
                return await self._forward(request, retry=False)

            if resp.status_code == 200 and is_exempt(request.method, request.url.path):
                self.mark_ready()

            async def body_iter():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    if cm is not None:
                        await cm.__aexit__(None, None, None)

            return StreamingResponse(
                body_iter(),
                status_code=resp.status_code,
                headers=_filter_resp_headers(resp.headers.items()),
            )
        except Exception:
            if cm is not None:
                await cm.__aexit__(None, None, None)
            raise

    def _build_app(self) -> Starlette:
        methods = [
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "HEAD",
            "OPTIONS",
        ]

        @asynccontextmanager
        async def lifespan(app: Starlette):
            await self.startup()
            task = asyncio.create_task(self.idle_loop(), name="vllm-idle-loop")
            try:
                yield
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                await self.shutdown()

        app = Starlette(
            routes=[
                Route("/", self.handle_proxy, methods=methods),
                Route("/{path:path}", self.handle_proxy, methods=methods),
            ],
            lifespan=lifespan,
        )
        app.state.proxy = self
        return app


def create_app(config: Config | None = None) -> ASGIApp:
    proxy = IdleProxy(config or Config.from_env())
    return proxy.app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [vllm-idle-proxy] %(message)s",
    )
    config = Config.from_env()
    proxy = IdleProxy(config)
    uvicorn.run(
        proxy.app,
        host=config.bind_host,
        port=config.bind_port,
        log_level="info",
        access_log=False,
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
