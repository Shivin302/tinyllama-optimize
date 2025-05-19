# TinyLlama Transformers Baseline

This is a baseline implementation for serving TinyLlama using the Hugging Face Transformers library with FastAPI.

## Features

- 🚀 FastAPI-based web server
- 🤗 Hugging Face Transformers integration
- 📊 Detailed generation metrics
- ⚡ Optimized for both CPU and GPU
- 🔄 Asynchronous request handling
- 🛡️ CORS enabled

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 4GB GPU memory (for 1B parameter model)

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:8001`

### API Endpoints

- `GET /health` - Health check endpoint
- `POST /generate` - Generate text from a prompt

### Example Requests

#### Health Check
```bash
curl http://localhost:8001/
```

#### Text Generation
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### Request Parameters

- `prompt` (str): The input text to generate from
- `max_tokens` (int, optional): Maximum number of tokens to generate. Defaults to 100.
- `temperature` (float, optional): Sampling temperature. Defaults to 0.7.
- `top_p` (float, optional): Nucleus sampling probability. Defaults to 0.9.
- `top_k` (int, optional): Top-k sampling. Defaults to 50.
- `repetition_penalty` (float, optional): Penalty for repeating tokens. Defaults to 1.1.
- `do_sample` (bool, optional): Whether to use sampling. Defaults to True.
- `num_return_sequences` (int, optional): Number of sequences to return. Defaults to 1.
- `stop` (List[str], optional): Stop sequences. Defaults to None.

## Testing

Run the test script to verify the API is working:

```bash
python test_api.py
```

## Performance Considerations

- The model is loaded in half-precision (FP16) when CUDA is available
- For better performance, use a GPU with at least 8GB of VRAM
- Adjust batch size and sequence length based on your hardware

## License

MIT
