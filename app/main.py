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
    # Initialize/load model & tokenizer on startup
    qwen_service.initialize()
    yield
    # Clean up executors
    if hasattr(qwen_service, "executor"):
        qwen_service.executor.shutdown(wait=True)

app = FastAPI(
    title="FastAPI AI Gateway",
    description="OpenAI compatible API backend powered by Qwen 2.5 1.5B Instruct",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Setup
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
    # Skip auth for docs, health endpoint, etc
    if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
        
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

# Error handlers mapping standard exceptions to OpenAI style JSON error envelopes
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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
