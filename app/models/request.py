from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (system, user, assistant, tool).")
    content: str = Field(..., description="The contents of the message.")
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use.")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation so far.")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=128)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None

class CompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use.")
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for.")
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(16, ge=1)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=128)
    stream: Optional[bool] = Field(False)
    logprobs: Optional[int] = None
    echo: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(1, ge=1)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
