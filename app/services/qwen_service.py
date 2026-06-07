import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings

class QwenService:
    """
    Singleton service class to manage loading the Qwen 2.5 1.5B Instruct model and 
    executing inference logic. It abstracts hardware management (CPU/GPU/MPS) and 
    prevents blocking the FastAPI async event loop by delegating computation to a ThreadPoolExecutor.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Enforce Singleton pattern so the model is only loaded into memory once.
        if not cls._instance:
            cls._instance = super(QwenService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        """
        Loads the model and tokenizer from Hugging Face Hub (or local directory)
        and configures optimization parameters based on the system's available hardware.
        """
        if self._initialized:
            return
        
        print(f"Loading Qwen model '{settings.MODEL_ID}' on device '{settings.DEVICE}'...")
        
        # 1. Resolve Device Target: Auto-detect NVIDIA GPU (cuda), Apple Silicon (mps), or fallback to CPU (cpu)
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
        
        # 2. Configure Model Precision: Use lower-precision types for GPU to reduce memory footprint and speed up inference
        torch_dtype = torch.float32
        if self.device in ["cuda", "mps"]:
            # Use bfloat16 if GPU architecture supports it, otherwise fallback to float16
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # 3. Load Tokenizer & Model Weights
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
        self.model.eval()  # Put model in evaluation mode (disables dropout layers, etc.)
        
        # 4. Initialize Thread Pool Executor
        # Deep learning inference is CPU/GPU bound and blocks execution.
        # Running it directly in an async function would block the entire FastAPI event loop,
        # stopping other incoming HTTP requests. We isolate it on a background thread.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = True
        print("Model loaded successfully.")

    def _sync_generate(self, input_ids: Any, generation_config: Dict[str, Any]) -> Tuple[str, int, int]:
        """
        Synchronous wrapper function that runs the model generation.
        Runs inside the ThreadPoolExecutor.
        """
        prompt_len = input_ids.shape[1]
        
        with torch.no_grad():  # Turn off gradient calculations to save memory and CPU cycles during inference
            outputs = self.model.generate(
                input_ids,
                **generation_config
            )
            
        # Extract only the newly generated token IDs (slice off the original prompt prefix)
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
        """
        Processes standard OpenAI list of messages, applies Qwen's specific Chat Template format,
        and schedules the inference on the ThreadPoolExecutor.
        """
        
        # Format the system/user/assistant message list into a single instruction string structured for Qwen
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Build the dynamic generation parameters
        generation_config = {
            "max_new_tokens": max_tokens or settings.MAX_NEW_TOKENS,
            "top_p": top_p,
            "do_sample": True if temperature > 0.0 else False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation_config["temperature"] = temperature
            
        # Get active async loop and delegate the blocking generation call to the background thread pool
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
        """
        Performs raw text completion (legacy endpoint logic) without applying a chat template wrapper.
        """
        
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

# Instantiate global service singleton
qwen_service = QwenService()
