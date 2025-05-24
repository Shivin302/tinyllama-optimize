# vllm_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams, AsyncLLMEngine
import uvicorn
import os
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext
import asyncio

app = FastAPI()

class GenerationRequest(BaseModel):
    prompts: List[str]
    request_id: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50


class vLLMServerDynamicBatching:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        llm_params = {
            "model": "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",  # Using AWQ quantized version
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
            "max_num_batched_tokens": 128 * 1024,
            "max_num_seqs": 128,
        }
        engine_args = AsyncEngineArgs(**llm_params)
        self.llm = AsyncLLMEngine.from_engine_args(engine_args, usage_context=UsageContext.API_SERVER)
        print("vLLM model loaded")

vllm_server = vLLMServerDynamicBatching()


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
        outputs_generator = vllm_server.llm.generate(
            request.prompts[0],
            sampling_params,
            request_id=request.request_id
        )
        
        final_output = None
        try:
            async for request_output in outputs_generator:
                final_output = request_output
        except asyncio.CancelledError:
            return Response(status_code=499)

        assert final_output is not None
        results = []
        for output in final_output.outputs:
            results.append({
                "text": output.text,
                "tokens": len(output.token_ids)
            })

        return {
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)