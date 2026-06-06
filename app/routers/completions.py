from fastapi import APIRouter, HTTPException
from app.models.request import CompletionRequest
from app.models.response import CompletionResponse, CompletionChoice, Usage
from app.services.qwen_service import qwen_service

router = APIRouter()

@router.post("/completions", response_model=CompletionResponse, tags=["Inference"])
async def completions(request: CompletionRequest):
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported in this version. Set stream=false."
        )

    # If prompt is a list, we handle the first one (standard simplicity)
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            raise HTTPException(status_code=400, detail="Empty prompt list received.")
        prompt_str = request.prompt[0]
    else:
        prompt_str = request.prompt

    try:
        completion_text, prompt_tokens, completion_tokens = await qwen_service.generate_completion(
            prompt=prompt_str,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 16
        )
        
        response = CompletionResponse(
            model=request.model,
            choices=[
                CompletionChoice(
                    text=completion_text,
                    index=0,
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
