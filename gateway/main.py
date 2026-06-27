from __future__ import annotations

import time
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gateway.config import settings
from gateway.logger import append_jsonl
from gateway.schemas import ChatCompletionRequest, ErrorBody, ErrorEnvelope

app = FastAPI(title="Phase 1 Gateway", version="0.1.0")


def build_error(
    *,
    message: str,
    error_type: str,
    code: str,
    request_id: str,
    status_code: int,
) -> JSONResponse:
    payload = ErrorEnvelope(
        error=ErrorBody(
            message=message,
            type=error_type,
            code=code,
            request_id=request_id,
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def extract_request_id(header_request_id: str | None) -> str:
    return header_request_id or str(uuid4())


def extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


async def log_request(
    *,
    request_id: str,
    path: str,
    model: str | None,
    max_tokens: int | None,
    status_code: int,
    latency_ms: float,
    error_type: str | None,
) -> None:
    append_jsonl(
        settings.gateway_log_path,
        {
            "request_id": request_id,
            "path": path,
            "model": model,
            "max_tokens": max_tokens,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "error_type": error_type,
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = extract_request_id(request.headers.get("x-request-id"))
    response = build_error(
        message="Request validation failed.",
        error_type="invalid_request_error",
        code="bad_request",
        request_id=request_id,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
    await log_request(
        request_id=request_id,
        path=str(request.url.path),
        model=None,
        max_tokens=None,
        status_code=response.status_code,
        latency_ms=0.0,
        error_type="invalid_request_error",
    )
    return response


@app.get("/healthz")
async def healthz() -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            upstream = await client.get(f"{settings.vllm_base_url}/health")
        if upstream.status_code >= 400:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "degraded", "upstream_status": upstream.status_code},
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "upstream_status": upstream.status_code},
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "down", "detail": str(exc)},
        )


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> JSONResponse:
    request_id = extract_request_id(x_request_id)
    start = time.perf_counter()
    token = extract_bearer_token(authorization)

    if token is None:
        response = build_error(
            message="Missing or invalid Authorization header.",
            error_type="authentication_error",
            code="unauthorized",
            request_id=request_id,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
        await log_request(
            request_id=request_id,
            path=str(request.url.path),
            model=payload.model,
            max_tokens=payload.max_tokens,
            status_code=response.status_code,
            latency_ms=(time.perf_counter() - start) * 1000,
            error_type="authentication_error",
        )
        return response

    if token != settings.gateway_token:
        response = build_error(
            message="Bearer token is invalid.",
            error_type="authentication_error",
            code="forbidden",
            request_id=request_id,
            status_code=status.HTTP_403_FORBIDDEN,
        )
        await log_request(
            request_id=request_id,
            path=str(request.url.path),
            model=payload.model,
            max_tokens=payload.max_tokens,
            status_code=response.status_code,
            latency_ms=(time.perf_counter() - start) * 1000,
            error_type="authentication_error",
        )
        return response

    body = payload.model_dump(mode="json", exclude_none=True)
    headers = {"X-Request-ID": request_id}

    try:
        async with httpx.AsyncClient(
            timeout=settings.gateway_timeout_seconds,
            trust_env=False,
        ) as client:
            upstream = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        response = build_error(
            message=f"Upstream vLLM request failed: {exc}",
            error_type="upstream_error",
            code="upstream_unavailable",
            request_id=request_id,
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
        await log_request(
            request_id=request_id,
            path=str(request.url.path),
            model=payload.model,
            max_tokens=payload.max_tokens,
            status_code=response.status_code,
            latency_ms=(time.perf_counter() - start) * 1000,
            error_type="upstream_error",
        )
        return response

    latency_ms = (time.perf_counter() - start) * 1000

    if upstream.status_code >= 400:
        message = upstream.text
        try:
            upstream_json = upstream.json()
            message = upstream_json.get("error", {}).get("message", message)
        except ValueError:
            pass

        response = build_error(
            message=message,
            error_type="invalid_request_error",
            code="upstream_error",
            request_id=request_id,
            status_code=upstream.status_code,
        )
        await log_request(
            request_id=request_id,
            path=str(request.url.path),
            model=payload.model,
            max_tokens=payload.max_tokens,
            status_code=response.status_code,
            latency_ms=latency_ms,
            error_type="invalid_request_error",
        )
        return response

    data = upstream.json()
    await log_request(
        request_id=request_id,
        path=str(request.url.path),
        model=payload.model,
        max_tokens=payload.max_tokens,
        status_code=upstream.status_code,
        latency_ms=latency_ms,
        error_type=None,
    )
    return JSONResponse(status_code=upstream.status_code, content=data, headers={"X-Request-ID": request_id})
