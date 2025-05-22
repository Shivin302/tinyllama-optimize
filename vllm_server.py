# vllm_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams
import uvicorn
import os

app = FastAPI()

class GenerationRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50

class vLLMServer:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        llm_params = {
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.8,
            "enforce_eager": True,
            "model": "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ"  # Using AWQ quantized version
        }
        self.llm = LLM(**llm_params)
        print("vLLM model loaded")

vllm_server = vLLMServer()

@app.post("/generate")
async def generate(request: GenerationRequest):
    if not vllm_server.llm:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens
    )
    
    try:
        outputs = vllm_server.llm.generate(
            request.prompts,
            sampling_params,
            use_tqdm=False
        )
        
        results = []
        for output in outputs:
            results.append({
                "text": output.outputs[0].text,
                "tokens": len(output.outputs[0].token_ids)
            })
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)