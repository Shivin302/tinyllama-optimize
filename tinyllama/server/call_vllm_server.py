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
from transformers import AutoTokenizer
from tinyllama.utils.llm_prompt import PROMPT

def truncate_prompt(prompt: str, max_length: int) -> str:
    """Truncate the prompt to the specified number of tokens using TinyLlama's tokenizer.
    
    Args:
        prompt: The input text prompt
        max_length: Maximum number of tokens to keep
        
    Returns:
        Truncated prompt as a string
    """
    # Initialize TinyLlama tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Encode the prompt to token IDs
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    # If the prompt is already shorter than max_length, return as is
    if len(token_ids) <= max_length:
        return prompt
        
    # Truncate the token IDs and decode back to text
    truncated_ids = token_ids[:max_length]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)

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
    client = vLLMClient(f"http://localhost:8000")
    
    # Test connection
    try:
        await client.generate(prompts=["Test"], max_tokens=1)
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return
    
    # Run profiling
    batch_sizes = [1, 2, 4]
    max_new_tokens_list = [64, 128]
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
    results.to_csv("vllm_server_profile.csv", index=False)
    print(f"\nProfiling results saved to vllm_server_profile.csv")
    print("\nProfiling Summary:")
    print(results[['batch_size', 'max_new_tokens', 
                  'avg_latency_ms', 'avg_tokens_per_second']].to_string())

if __name__ == "__main__":
    asyncio.run(main())