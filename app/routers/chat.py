from fastapi import APIRouter, HTTPException
from app.models.request import ChatCompletionRequest
from app.models.response import ChatCompletionResponse, ChatCompletionChoice, ChatMessageResponse, Usage
from app.services.qwen_service import qwen_service
from app.config import settings

router = APIRouter()

@router.post("/chat/completions", response_model=ChatCompletionResponse, tags=["Inference"])
async def chat_completions(request: ChatCompletionRequest):
    # Streaming is not implemented in this version, return 400 if requested
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming responses are not supported in this version. Set stream=false."
        )

    # Convert request messages to format expected by tokenizer
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    
    try:
        completion_text, prompt_tokens, completion_tokens = await qwen_service.generate_chat(
            messages=messages_dict,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        # Check if the user wanted JSON format specifically
        # We process/ensure format if they asked or requested structured JSON
        # Here we map directly to the OpenAI structure
        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageResponse(role="assistant", content=completion_text),
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
