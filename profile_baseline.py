"""
TinyLlama Model Profiling Script

This script profiles the TinyLlama model to analyze its performance characteristics
across different batch sizes, sequence lengths, and generation configurations.
"""

import os
import time
import torch
import psutil
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import gc
from datetime import datetime
from utils.llm_prompt import PROMPT
from utils.profile import ModelProfiler

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineProfiler(ModelProfiler):
    def __init__(self, model_name: str, device: str = None):
        """Initialize the profiler with the model and tokenizer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_model()
        

    def load_model(self):
        print(f"Loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded on {next(self.model.parameters()).device}")
    
    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """Truncate the prompt to the specified number of tokens."""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)[0]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return prompt

    def profile_generation(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        batch_size: int = 1,
        input_length: int = 128,
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> Dict[str, float]:
        """Profile text generation with the given parameters.
        
        Args:
            prompt: The input prompt text
            max_new_tokens: Maximum number of tokens to generate
            batch_size: Number of sequences to generate in parallel
            input_length: Desired input length in tokens (will truncate if needed)
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs before profiling
            
        Returns:
            Dictionary containing profiling metrics
        """
        
        # Prepare inputs
        inputs = self.tokenizer([prompt] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        self.clear_cuda_cache()
        
        # Profile runs
        latencies = []
        tokens_per_second = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            # Calculate metrics
            duration = time.time() - start_time
            # Only count newly generated tokens (after input_length)
            num_generated_tokens = (outputs[0, inputs['input_ids'].shape[1]:] != self.tokenizer.eos_token_id).sum().item()
            
            latencies.append(duration)
            tokens_per_second.append(num_generated_tokens / duration)
        
        # Calculate statistics
        stats = {
            'batch_size': batch_size,
            'input_length': input_length,
            'max_new_tokens': max_new_tokens,
            'avg_latency_ms': round(np.mean(latencies) * 1000, 2),
            'std_latency_ms': round(np.std(latencies) * 1000, 2),
            'avg_tokens_per_second': round(np.mean(tokens_per_second), 2),
            'std_tokens_per_second': round(np.std(tokens_per_second), 2),
            'memory_usage_mb': round(self.get_memory_usage()['rss_mb'], 2),
        }
        
        if torch.cuda.is_available():
            stats['cuda_max_memory_allocated_mb'] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
            torch.cuda.reset_peak_memory_stats()
        
        return stats
    

def main():
    """Main function to run the profiling."""
    print("Starting TinyLlama model profiling...")
    print(f"Using device: {DEVICE}")
    
    profiler = BaselineProfiler(MODEL_NAME, DEVICE)
    
    print("\nStarting comprehensive profiling...")
    results = profiler.run_comprehensive_profile(
        num_runs=3,
        warmup_runs=1,
        output_dir="baseline_profiling_results"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "baseline_profiling_results"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = f"{output_dir}/results_{timestamp}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nProfiling results saved to: {csv_path}")
    
    profiler.plot_results(results, output_dir)
    print("Generated plots in the baseline_profiling_results directory.")
    
    print("\nProfiling Summary:")
    print(results[['batch_size', 'max_new_tokens', 
                    'avg_latency_ms', 'avg_tokens_per_second']].to_string())
        

if __name__ == "__main__":
    main()
