from fastapi import APIRouter
from app.models.response import ModelListResponse, ModelCard
from app.config import settings

router = APIRouter()

@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}

@router.get("/models", response_model=ModelListResponse, tags=["Models"])
async def list_models():
    model_name = settings.MODEL_ID.split("/")[-1].lower()
    return ModelListResponse(
        data=[
            ModelCard(id=model_name, owned_by="qwen")
        ]
    )
