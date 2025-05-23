"""
TinyLlama vLLM Profiling Script

This script profiles the TinyLlama model using the vLLM library to analyze its
performance characteristics across different batch sizes and sequence lengths.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime
from vllm import LLM, SamplingParams
from tinyllama.utils.llm_prompt import PROMPT
from tinyllama.utils.profile import ModelProfiler
import torch


# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class vLLMProfiler(ModelProfiler):
    def __init__(self, model_name: str):
        """Initialize the vLLM profiler with the model."""
        super().__init__(model_name, DEVICE)
    
    def load_model(self):
        print(f"Loading model {self.model_name} with vLLM...")

        llm_params = {
            "model": self.model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
        }

        kv_cache = False
        if kv_cache:
            llm_params["kv_cache_dtype"] = "fp8_e5m2"
            llm_params["calculate_kv_scales"] = True

        awq_4bit = True
        if awq_4bit:
            llm_params["model"] = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ"

        prefix_caching = False
        if prefix_caching:
            llm_params["enable_prefix_caching"] = True

        speculative_decoding = False
        if speculative_decoding:
            llm_params["speculative_config"] = {
                "model": "yuhuili/EAGLE-llama2-chat-7B",
                "num_speculative_tokens": 5,
            }


        self.llm = LLM(**llm_params)
        print(f"vLLM model loaded on {DEVICE}")
    

    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """Truncate the prompt to the specified number of tokens.
        
        Args:
            prompt: The input text prompt
            max_length: Maximum number of tokens to keep
            
        Returns:
            Truncated prompt as a string
        """
        # Get the tokenizer from the vLLM model
        tokenizer = self.llm.get_tokenizer()
        
        # Encode the prompt to token IDs
        token_ids = tokenizer.encode(prompt)
        
        # If the prompt is already shorter than max_length, return as is
        if len(token_ids) <= max_length:
            return prompt
            
        # Truncate the token IDs and decode back to text
        truncated_ids = token_ids[:max_length]
        return tokenizer.decode(truncated_ids)


    def profile_generation(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        batch_size: int = 1,
        input_length: int = 128,
        num_runs: int = 3,
        warmup_runs: int = 1
    ) -> Dict[str, float]:
        """Profile text generation with the given parameters."""
        # Truncate prompt to desired input length
        prompt = self._truncate_prompt(prompt, input_length)
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=0.0,  # Disable sampling for deterministic benchmarking
            max_tokens=max_new_tokens,
        )
        
        # Prepare inputs
        prompts = [prompt] * batch_size
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = self.llm.generate(prompts, sampling_params)

        self.clear_cuda_cache()
        
        # Profile runs
        latencies = []
        tokens_per_second = []
        
        for _ in range(num_runs):
            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            duration = time.time() - start_time
            
            # Calculate metrics
            num_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            avg_tokens_per_second = num_output_tokens / duration
            
            latencies.append(duration)
            tokens_per_second.append(avg_tokens_per_second)
        
        stats = {
            'batch_size': batch_size,
            'input_length': input_length,
            'max_new_tokens': max_new_tokens,
            'avg_latency_ms': round(np.mean(latencies) * 1000, 2),
            'std_latency_ms': round(np.std(latencies) * 1000, 2),
            'avg_tokens_per_second': round(np.mean(tokens_per_second), 2),
            'std_tokens_per_second': round(np.std(tokens_per_second), 2),
        }
        
        return stats
    

def main():
    print("Starting TinyLlama vLLM profiling...")
    print(f"Using device: {DEVICE}")
    
    profiler = vLLMProfiler(MODEL_NAME)
    
    print("\nStarting comprehensive profiling...")
    results = profiler.run_comprehensive_profile(
        num_runs=3,
        warmup_runs=1,
    )
            
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "profiling_results/vllm"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = f"{output_dir}/results_{timestamp}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nProfiling results saved to: {csv_path}")
    
    
    print("\nProfiling Summary:")
    print(results[['batch_size', 'max_new_tokens', 
                    'avg_latency_ms', 'avg_tokens_per_second']].to_string())

if __name__ == "__main__":
    main()
