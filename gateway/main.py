"""
Phase 1 Gateway — FastAPI 网关服务
===================================
作为 vLLM 推理服务的前置代理，负责：
- 统一认证（Bearer token 校验）
- 请求/响应格式转换与错误封装
- 请求日志记录（JSONL 格式）
- 健康检查代理

对外暴露的端点：
  GET  /healthz              — 健康检查（含上游探活）
  POST /v1/chat/completions  — 聊天补全（兼容 OpenAI API 格式）
"""

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

# ---- 应用实例 ----
app = FastAPI(title="Phase 1 Gateway", version="0.1.0")


# ---- 工具函数 ----

def build_error(
    *,
    message: str,
    error_type: str,
    code: str,
    request_id: str,
    status_code: int,
) -> JSONResponse:
    """
    构建统一格式的错误响应。

    所有参数均为 keyword-only（* 强制），避免调用时参数顺序出错。

    参数:
        message:     人类可读的错误描述（如 "Bearer token is invalid."）
        error_type:  错误大类（如 "authentication_error"、"upstream_error"）
        code:        错误细粒度码（如 "unauthorized"、"forbidden"）
        request_id:  请求追踪 ID，来自 X-Request-ID 头或自动生成
        status_code: HTTP 状态码

    返回:
        JSONResponse，body 结构为 { "error": { "message", "type", "code", "request_id" } }
    """
    payload = ErrorEnvelope(
        error=ErrorBody(
            message=message,
            type=error_type,
            code=code,
            request_id=request_id,
        )
    )
    # model_dump() 将 Pydantic 模型转为 dict，再由 JSONResponse 序列化
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def extract_request_id(header_request_id: str | None) -> str:
    """
    从请求头中提取 request_id，若无则自动生成一个 UUID4。

    参数:
        header_request_id: X-Request-ID 请求头的值（可能为 None）

    返回:
        request_id 字符串
    """
    return header_request_id or str(uuid4())


def extract_bearer_token(authorization: str | None) -> str | None:
    """
    从 Authorization 头中提取 Bearer token。

    参数:
        authorization: Authorization 请求头的原始值

    返回:
        token 字符串；若格式不正确或缺失则返回 None
    """
    if not authorization:
        return None
    # "Bearer <token>" → ["Bearer", "", "<token>"]
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
    """
    将一次请求的关键字段追加写入 JSONL 日志文件。

    日志路径由 settings.gateway_log_path 配置。
    所有字段均为 keyword-only，避免调用时传参错位。
    """
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


# ---- 异常处理器 ----

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    捕获 FastAPI 的请求校验错误（如缺少必填字段、类型不匹配），
    统一包装为 422 错误响应，并记录日志。
    """
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


# ---- 健康检查端点 ----

@app.get("/healthz")
async def healthz() -> JSONResponse:
    """
    健康检查端点。

    探测上游 vLLM 的 /health 接口：
    - 上游正常 → 200 { "status": "ok" }
    - 上游异常（≥400） → 503 { "status": "degraded" }
    - 上游不可达（HTTPError） → 503 { "status": "down" }
    """
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


# ---- 聊天补全端点 ----

@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
) -> JSONResponse:
    """
    OpenAI 兼容的聊天补全端点。

    请求流程:
    1. 提取/生成 request_id（区别不同request链路），开始计时
    2. 校验 Bearer token：
       - 缺失 → 401
       - 与配置不匹配 → 403
    3. 将请求体透传给上游 vLLM 的 /v1/chat/completions
    4. 上游不可达 → 502
    5. 上游返回 ≥400 → 透传状态码并包装错误信息
    6. 上游正常 → 透传响应体

    所有分支均记录 JSONL 日志。
    """
    # ---- 1. 初始化 ----
    request_id = extract_request_id(x_request_id)
    start = time.perf_counter()
    token = extract_bearer_token(authorization)

    # ---- 2a. 缺少 Authorization 头 → 401 ----
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

    # ---- 2b. token 与配置不匹配 → 403 ----
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

    # ---- 3. 构造上游请求 ----
    # model_dump(mode="json") 确保 datetime/UUID 等字段正确序列化
    # exclude_none=True 去掉 None 值字段，避免上游误解析
    body = payload.model_dump(mode="json", exclude_none=True)
    headers = {"X-Request-ID": request_id}

    try:
        async with httpx.AsyncClient( # 异步请求上游
            timeout=settings.gateway_timeout_seconds,
            trust_env=False,
        ) as client:
            upstream = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
    # ---- 4. 上游不可达 → 502 ----
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

    # ---- 5. 上游返回错误（≥400）→ 透传并包装 ----
    if upstream.status_code >= 400:
        # 优先提取上游 JSON 中的 error.message，如果不行就回退到原始文本
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

    # ---- 6. 上游正常 → 透传响应 ----
    data = upstream.json()
    await log_request(
        request_id=request_id,
        path=str(request.url.path),
        model=payload.model,
        max_tokens=payload.max_tokens,
        status_code=upstream.status_code,
        latency_ms=latency_ms,
        error_type=None,  # 正常请求无错误类型
    )
    return JSONResponse(
        status_code=upstream.status_code,
        content=data,
        headers={"X-Request-ID": request_id},
    )
