import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time
import logging
import os
from model import load_model, DEVICE, MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TinyLlama Transformers API",
             description="Transformers-based API for TinyLlama inference")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, tokenizer = load_model()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    stop: Optional[List[str]] = None

class TokenMetrics(BaseModel):
    prompt_tokens: int
    generated_tokens: int
    total_time: float
    tokens_per_second: float

class GenerationResponse(BaseModel):
    generated_text: str
    metrics: TokenMetrics

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "model": MODEL_NAME
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        generation_config = {
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
            "num_return_sequences": request.num_return_sequences,
        }
        
        result = generate_text(
            prompt=request.prompt,
            generation_config=generation_config,
            stop=request.stop
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_text(prompt: str, generation_config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate text using the model with the given parameters."""
    # Encode the input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs.input_ids.shape[1]
    
    # Generate text
    start_time = time.time()
    
    outputs = model.generate(
        **inputs,
        **generation_config,
        pad_token_id=tokenizer.eos_token_id,
        )
        
    # Calculate metrics
    end_time = time.time()
    total_time = end_time - start_time
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    
    return {
        "generated_text": generated_text,
        "metrics": {
            "prompt_tokens": input_length,
            "generated_tokens": len(generated_tokens),
            "total_time": total_time,
            "tokens_per_second": len(generated_tokens) / total_time if total_time > 0 else 0
        }
    }




if __name__ == "__main__":
    import uvicorn
    import sys
    
    dev_mode = True
    host = "0.0.0.0"
    port = 8001
    
    if dev_mode and len(sys.argv) > 1 and sys.argv[1] == "--reload":
        # For development with auto-reload
        uvicorn.run("app:app", host=host, port=port, reload=True)
    else:
        # For production or direct execution
        uvicorn.run(app, host=host, port=port)