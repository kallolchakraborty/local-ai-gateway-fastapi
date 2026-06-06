# FastAPI AI Gateway (Qwen 2.5 1.5B)

A FastAPI implementation of the OpenAI chat completions API specification, serving a locally run Qwen 2.5 1.5B Instruct model via HuggingFace Transformers.

## Features

- **OpenAI-Compatible endpoints**: Plugs directly into standard OpenAI clients (SDKs, LangChain, Autogen, LiteLLM) by overriding the `base_url`.
- **Automatic Device Routing**: Auto-detects and loads the model on Apple Silicon (MPS), NVIDIA CUDA GPUs, or standard CPU.
- **Robust Schema Design**: Fully validating JSON request and response payloads built on top of Pydantic v2.
- **Clean Architecture**: Multi-threaded execution isolation for Hugging Face inference logic so it does not block the FastAPI async event loop.

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure (Optional)**:
   Modify the values inside `.env` to switch parameters like `PORT`, `API_KEY` authentication, or specify a custom device.

   * **Model Download Behavior**: By default, the application will automatically download the Qwen 2.5 1.5B Instruct model (approx. 3GB) from Hugging Face Hub on the first startup. Subsequent runs will use the cached local copy offline.
   * **Custom Local Model Path**: If you already have the model weights downloaded locally, you can change the `MODEL_ID` in `.env` to point to your absolute directory path (e.g. `MODEL_ID=/Users/username/models/Qwen2.5-1.5B-Instruct`) to load it instantly without any downloads.

3. **Start the server**:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

## Example API Queries

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Models List
```bash
curl http://127.0.0.1:8000/v1/models
```

### Chat Completions
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a joke about programming."}
    ],
    "temperature": 0.7
  }'
```
