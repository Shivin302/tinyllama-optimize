# profile_vllm_server.py
import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import os
from utils.llm_prompt import PROMPT
from vllm_server import vllm_server
from vllm import LLM


llm_params = {
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.05,
    "enforce_eager": True,
    "model": "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ"  # Using AWQ quantized version
}
llm = LLM(**llm_params)

def truncate_prompt(prompt: str, max_length: int) -> str:
    """Truncate the prompt to the specified number of tokens.
    
    Args:
        prompt: The input text prompt
        max_length: Maximum number of tokens to keep
        
    Returns:
        Truncated prompt as a string
    """
    # Get the tokenizer from the vLLM model
    tokenizer = llm.get_tokenizer()
    
    # Encode the prompt to token IDs
    token_ids = tokenizer.encode(prompt)
    
    # If the prompt is already shorter than max_length, return as is
    if len(token_ids) <= max_length:
        return prompt
        
    # Truncate the token IDs and decode back to text
    truncated_ids = token_ids[:max_length]
    return tokenizer.decode(truncated_ids)

class vLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    async def generate(
        self,
        prompts: List[str],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/generate"
        payload = {
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error: {error}")
                return await response.json()

async def profile_generation(
    client: vLLMClient,
    prompt: str,
    batch_sizes: List[int],
    max_new_tokens_list: List[int],
    num_runs: int = 3,
    warmup_runs: int = 1
) -> pd.DataFrame:
    results = []
    
    for batch_size in tqdm(batch_sizes, desc="Batch sizes"):
        for max_tokens in max_new_tokens_list:
            # Warmup runs
            for _ in range(warmup_runs):
                await client.generate(
                    prompts=[prompt] * batch_size,
                    max_tokens=max_tokens
                )
            
            # Profile runs
            latencies = []
            tokens_per_second = []
            
            for _ in range(num_runs):
                start_time = time.time()
                response = await client.generate(
                    prompts=[prompt] * batch_size,
                    max_tokens=max_tokens
                )
                duration = time.time() - start_time
                
                # Calculate metrics
                total_tokens = sum(r["tokens"] for r in response["results"])
                avg_tokens_per_second = total_tokens / duration
                
                latencies.append(duration * 1000)  # Convert to ms
                tokens_per_second.append(avg_tokens_per_second)
            
            # Record stats
            results.append({
                "batch_size": batch_size,
                "max_new_tokens": max_tokens,
                "avg_latency_ms": round(np.mean(latencies), 2),
                "std_latency_ms": round(np.std(latencies), 2),
                "avg_tokens_per_second": round(np.mean(tokens_per_second), 2),
                "std_tokens_per_second": round(np.std(tokens_per_second), 2),
            })
    
    return pd.DataFrame(results)

async def main():
    parser = argparse.ArgumentParser(description="Profile vLLM server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--output", type=str, default="vllm_server_profile.csv", 
                       help="Output CSV file")
    args = parser.parse_args()
    
    client = vLLMClient(f"http://{args.host}:{args.port}")
    
    # Test connection
    try:
        await client.generate(prompts=["Test"], max_tokens=1)
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return
    
    # Run profiling
    batch_sizes = [1, 2, 4, 8, 16]
    max_new_tokens_list = [32, 64, 128]
    input_length = 64

    prompt = truncate_prompt(PROMPT, input_length)
    
    print("Starting vLLM server profiling...")
    results = await profile_generation(
        client=client,
        prompt=prompt,
        batch_sizes=batch_sizes,
        max_new_tokens_list=max_new_tokens_list,
        num_runs=3,
        warmup_runs=1
    )
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nProfiling results saved to {args.output}")
    print("\nProfiling Summary:")
    print(results[['batch_size', 'max_new_tokens', 
                  'avg_latency_ms', 'avg_tokens_per_second']].to_string())

if __name__ == "__main__":
    asyncio.run(main())