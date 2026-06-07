from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.services.qwen_service import qwen_service
from app.routers import chat, completions, health
from app.models.response import OpenAIErrorResponse, OpenAIErrorDetail

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan management. Handles code execution blocks that must run
    during application startup (loading weights) and shutdown (clean execution pools).
    """
    # Initialize and pre-load Qwen model weights and tokenizer into RAM/VRAM
    qwen_service.initialize()
    yield
    # Clean up and gracefully terminate thread executor pool to prevent resource leaks
    if hasattr(qwen_service, "executor"):
        qwen_service.executor.shutdown(wait=True)

app = FastAPI(
    title="FastAPI AI Gateway",
    description="OpenAI compatible API backend powered by Qwen 2.5 1.5B Instruct",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Setup - Enables external API query access from browser clients (e.g. playground applications)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Bearer Token Security Middleware
@app.middleware("http")
async def verify_auth_token(request: Request, call_next):
    """
    Custom HTTP Middleware. Intercepts incoming requests to validate authentication.
    If API_KEY is defined in config, requests must present Header: 'Authorization: Bearer <API_KEY>'
    """
    # 1. Skip token authentication checks for public endpoints
    if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
        
    # 2. Enforce Bearer Token check if configured in .env properties
    if settings.API_KEY:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=OpenAIErrorResponse(
                    error=OpenAIErrorDetail(
                        message="Missing or invalid API Key in Authorization header.",
                        type="invalid_request_error",
                        code="invalid_api_key"
                    )
                ).model_dump()
            )
        token = auth_header.split(" ")[1]
        if token != settings.API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content=OpenAIErrorResponse(
                    error=OpenAIErrorDetail(
                        message="Invalid Authentication Token.",
                        type="invalid_request_error",
                        code="invalid_api_key"
                    )
                ).model_dump()
            )
            
    return await call_next(request)

# Error Handlers mapping internal exceptions to OpenAI JSON Error format specs
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles request formatting & Pydantic validation errors,
    re-mapping them to OpenAI style 'invalid_request_error' objects.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=OpenAIErrorResponse(
            error=OpenAIErrorDetail(
                message=str(exc.errors()),
                type="invalid_request_error",
                param=str(exc.errors()[0]["loc"]) if exc.errors() else None,
                code="validation_error"
            )
        ).model_dump()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all internal server exceptions, returning a normalized OpenAI error payload
    instead of exposing standard raw Python traceback dumps.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=OpenAIErrorResponse(
            error=OpenAIErrorDetail(
                message=str(exc),
                type="api_error",
                code="internal_error"
            )
        ).model_dump()
    )

# Include Routers
app.include_router(health.router)
app.include_router(chat.router, prefix="/v1")
app.include_router(completions.router, prefix="/v1")
