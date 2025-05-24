import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from tinyllama.utils.llm_prompt import PROMPT


CONFIG = {
    "host": "localhost",
    "port": 8000,
    "output_file": "vllm_concurrent_profile.csv",
    "batch_sizes": [1],
    "max_tokens_list": [512],
    "concurrency_levels": [1, 2, 4, 8, 16, 32],
    "requests_per_level": 128,
    "input_length": 128
}

class vLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = aiohttp.ClientSession()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
    async def generate(
        self,
        prompts: List[str],
        request_id: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/generate"
        payload = {
            "prompts": prompts,
            "request_id": request_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        
        try:
            start_time = time.time()
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API error: {error}")
                result = await response.json()
            latency = time.time() - start_time
            return latency * 1000, result
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")


def truncate_prompt(prompt: str, max_length: int) -> str:
    """Truncate the prompt to the specified number of tokens using TinyLlama's tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= max_length:
        return prompt
    return tokenizer.decode(token_ids[:max_length], skip_special_tokens=True)

async def make_request(
    client: vLLMClient,
    prompt: str,
    batch_size: int,
    max_tokens: int
) -> Tuple[float, float]:
    """Make a single request and return latency and tokens per second."""
    request_id = str(time.time())
    # print(f"Sending request: {request_id}")
    latency, response = await client.generate(
        prompts=[prompt] * batch_size,
        max_tokens=max_tokens,
        request_id=request_id
    )
    # print(f"Received request: {request_id}")
    # print("Prompt: ", prompt)
    # print("Response: ", response["results"][0]["text"])
    # print(response["results"][0]["tokens"])
    total_tokens = sum(r["tokens"] for r in response["results"])
    return latency, total_tokens

async def profile_concurrent_requests(
    client: vLLMClient,
    prompt: str,
    batch_size: int,
    max_tokens: int,
    num_requests: int,
    concurrency: int
) -> Dict[str, float]:
    """Profile concurrent requests and return aggregated metrics."""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def make_request_with_semaphore():
        async with semaphore:
            latency, total_tokens = await make_request(client, prompt, batch_size, max_tokens)

        return latency, total_tokens
    
    # Warmup
    for _ in range(5):
        await make_request(client, prompt, batch_size, max_tokens)
    

    # Run concurrent requests
    total_time = 0
    start_time = time.time()
    tasks = [make_request_with_semaphore() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    total_time += time.time() - start_time
    
    latencies, num_tokens = zip(*results)
    return {
        "concurrency": concurrency,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "total_requests": num_requests,
        "avg_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "p90_latency_ms": np.percentile(latencies, 90),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_tps": np.sum(num_tokens) / total_time,
    }

async def run_profiling(
    client: vLLMClient,
    prompt: str,
    batch_sizes: List[int],
    max_tokens_list: List[int],
    concurrency_levels: List[int],
    requests_per_level: int
) -> pd.DataFrame:
    """Run profiling with different concurrency levels."""
    results = []
    
    # Outer progress bar for batch sizes
    batch_bar = tqdm(batch_sizes, desc="Batch Progress", position=0)
    for batch_size in batch_bar:
        batch_bar.set_description(f"Batch Size: {batch_size}")
        
        # Middle progress bar for max tokens
        tokens_bar = tqdm(max_tokens_list, desc="Tokens Progress", position=1, leave=False)
        for max_tokens in tokens_bar:
            tokens_bar.set_description(f"Max Tokens: {max_tokens}")
            
            # Inner progress bar for concurrency levels
            concurrency_bar = tqdm(concurrency_levels, desc="Concurrency Progress", position=2, leave=False)
            for concurrency in concurrency_bar:
                concurrency_bar.set_description(f"Concurrency: {concurrency}")
                
                metrics = await profile_concurrent_requests(
                    client=client,
                    prompt=prompt,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    num_requests=requests_per_level,
                    concurrency=concurrency
                )
                results.append(metrics)
                
            concurrency_bar.close()
        tokens_bar.close()
    batch_bar.close()
    
    return pd.DataFrame(results)


async def main():
    print("Starting vLLM server profiling with configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    async with vLLMClient(f"http://{CONFIG['host']}:{CONFIG['port']}") as client:
        try:            
            print("Testing generate endpoint...")
            response = await client.generate(prompts=["Test"], max_tokens=1, request_id="test")
            print("Generate endpoint test successful")
            
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return
        
        # Prepare prompt
        prompt = truncate_prompt(PROMPT, CONFIG["input_length"])
        
        print("\nStarting profiling...")
        
        results = await run_profiling(
            client=client,
            prompt=prompt,
            batch_sizes=CONFIG["batch_sizes"],
            max_tokens_list=CONFIG["max_tokens_list"],
            concurrency_levels=CONFIG["concurrency_levels"],
            requests_per_level=CONFIG["requests_per_level"]
        )
        
        results["input_length"] = CONFIG["input_length"]

        
        # Format all float columns to 2 decimal places
        float_cols = results.select_dtypes(include=['float64']).columns
        results[float_cols] = results[float_cols].round(2)
        
        # Save results with formatted numbers
        results.to_csv(CONFIG["output_file"], index=False, float_format='%.2f')
        print(f"\nProfiling results saved to {CONFIG['output_file']}")
        
        # Format and print summary
        pd.set_option('display.float_format', '{:.2f}'.format)
        print("\nProfiling Summary:")
        
        # Define columns to display in summary
        summary_columns = [
            'input_length', 'max_tokens', 'concurrency', 'batch_size',
            'avg_latency_ms', 'throughput_tps',
        ]
        
        print(results[summary_columns].to_string())
        
        # Print best configuration
        best_throughput = results['throughput_tps'].max()
        best_latency = results[results['concurrency'] == 1]['avg_latency_ms'].min()
        
        print(f"\nBest throughput: {best_throughput:.2f} tokens/second")
        print(f"Best latency (concurrency=1): {best_latency:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
