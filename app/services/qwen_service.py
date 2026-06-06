import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings

class QwenService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(QwenService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        
        print(f"Loading Qwen model '{settings.MODEL_ID}' on device '{settings.DEVICE}'...")
        
        # Resolve device
        if settings.DEVICE == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = settings.DEVICE
            
        print(f"Resolved device: {self.device}")
        
        # Handle float16/bfloat16/float32 options based on device
        torch_dtype = torch.float32
        if self.device in ["cuda", "mps"]:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_ID, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Thread pool to isolate blocking model execution from FastAPI event loop
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = True
        print("Model loaded successfully.")

    def _sync_generate(self, input_ids: Any, generation_config: Dict[str, Any]) -> Tuple[str, int, int]:
        prompt_len = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                **generation_config
            )
            
        generated_ids = outputs[0][prompt_len:]
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        completion_len = len(generated_ids)
        total_len = prompt_len + completion_len
        
        return completion, prompt_len, completion_len

    async def generate_chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        top_p: float = 1.0, 
        max_tokens: int = None
    ) -> Tuple[str, int, int]:
        
        # Format input using Chat Template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generation_config = {
            "max_new_tokens": max_tokens or settings.MAX_NEW_TOKENS,
            "top_p": top_p,
            "do_sample": True if temperature > 0.0 else False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_config["temperature"] = temperature
            
        loop = asyncio.get_running_loop()
        completion, prompt_tokens, completion_tokens = await loop.run_in_executor(
            self.executor,
            self._sync_generate,
            model_inputs.input_ids,
            generation_config
        )
        
        return completion, prompt_tokens, completion_tokens

    async def generate_completion(
        self, 
        prompt: str, 
        temperature: float = 1.0, 
        top_p: float = 1.0, 
        max_tokens: int = 16
    ) -> Tuple[str, int, int]:
        
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        generation_config = {
            "max_new_tokens": max_tokens,
            "top_p": top_p,
            "do_sample": True if temperature > 0.0 else False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_config["temperature"] = temperature
            
        loop = asyncio.get_running_loop()
        completion, prompt_tokens, completion_tokens = await loop.run_in_executor(
            self.executor,
            self._sync_generate,
            model_inputs.input_ids,
            generation_config
        )
        
        return completion, prompt_tokens, completion_tokens

qwen_service = QwenService()
