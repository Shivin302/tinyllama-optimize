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
from abc import ABC, abstractmethod

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test configurations
BATCH_SIZES = [1, 2]
MAX_NEW_TOKENS = [64, 128]  # Test different output lengths
INPUT_LENGTHS = [64, 128]


class ModelProfiler(ABC):
    def __init__(self, model_name: str, device: str = None):
        """Initialize the profiler with the model and tokenizer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_model()
        
    @abstractmethod
    def load_model(self):
        pass
    
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    def clear_cuda_cache(self):
        """Clear CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _truncate_prompt(self):
        pass
    

    @abstractmethod
    def profile_generation(self):
        pass


    def run_comprehensive_profile(
        self,
        num_runs: int = 3,
        warmup_runs: int = 1,
        output_dir: str = "profiling_results"
    ) -> pd.DataFrame:
        """Run a comprehensive profiling of the model with different configurations.
        
        Args:
            prompt: Input prompt text
            batch_sizes: List of batch sizes to test
            max_new_tokens_list: List of output lengths to test
            input_lengths: List of input lengths to test (in tokens)
            num_runs: Number of profiling runs per configuration
            warmup_runs: Number of warmup runs before profiling
            output_dir: Directory to save results
            
        Returns:
            DataFrame containing profiling results
        """        
        output_dir = os.path.join(os.path.dirname(__file__), "profiling_results")
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        total_combinations = len(BATCH_SIZES) * len(MAX_NEW_TOKENS) * len(INPUT_LENGTHS)
        
        with tqdm(total=total_combinations, desc="Profiling") as pbar:
            for batch_size in BATCH_SIZES:
                for max_new_tokens in MAX_NEW_TOKENS:
                    for input_length in INPUT_LENGTHS:
                        try:
                            # Profile with current configuration
                            truncated_prompt = self._truncate_prompt(PROMPT, input_length)
                            stats = self.profile_generation(
                                prompt=truncated_prompt,
                                max_new_tokens=max_new_tokens,
                                batch_size=batch_size,
                                input_length=input_length,
                                num_runs=num_runs,
                                warmup_runs=warmup_runs
                            )
                            results.append(stats)
                            
                            # Print progress
                            pbar.set_postfix({
                                'b': batch_size,
                                'in': input_length,
                                'out': max_new_tokens,
                                # 'latency': f"{stats['avg_latency_ms']:.1f}ms"
                            })
                            
                        except Exception as e:
                            print(f"\nError with batch={batch_size}, in_len={input_length}, out_len={max_new_tokens}: {str(e)}")
                            self.clear_cuda_cache()
                            time.sleep(1)  # Give some time to recover
                        
                        pbar.update(1)
        
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"{output_dir}/profiling_results_{timestamp}.csv", index=False)
        return df
    
    @staticmethod
    def plot_results(df: pd.DataFrame, output_dir: str = "profiling_results"):
        """Plot profiling results with support for different input lengths."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot latency vs batch size for different input lengths and output lengths
        plt.figure(figsize=(14, 8))
        
        # Create a grid of subplots: one row per input length
        fig, axes = plt.subplots(len(df['input_length'].unique()), 1, 
                               figsize=(14, 6 * len(df['input_length'].unique())))
        if len(df['input_length'].unique()) == 1:
            axes = [axes]  # Ensure axes is always a list
            
        for ax, (input_len, group) in zip(axes, df.groupby('input_length')):
            for max_tokens in sorted(group['max_new_tokens'].unique()):
                subset = group[group['max_new_tokens'] == max_tokens]
                ax.errorbar(
                    subset['batch_size'],
                    subset['avg_latency_ms'],
                    yerr=subset['std_latency_ms'],
                    label=f'Output={max_tokens} tokens',
                    marker='o',
                    capsize=5
                )
            
            ax.set_title(f'Input Length: {input_len} tokens')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Latency (ms)')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_vs_batch_size.png")
        plt.close()
        
        # Plot tokens per second vs batch size
        fig, axes = plt.subplots(len(df['input_length'].unique()), 1, 
                               figsize=(14, 6 * len(df['input_length'].unique())))
        if len(df['input_length'].unique()) == 1:
            axes = [axes]  # Ensure axes is always a list
            
        for ax, (input_len, group) in zip(axes, df.groupby('input_length')):
            for max_tokens in sorted(group['max_new_tokens'].unique()):
                subset = group[group['max_new_tokens'] == max_tokens]
                ax.errorbar(
                    subset['batch_size'],
                    subset['avg_tokens_per_second'],
                    yerr=subset['std_tokens_per_second'],
                    label=f'Output={max_tokens} tokens',
                    marker='o',
                    capsize=5
                )
            
            ax.set_title(f'Input Length: {input_len} tokens')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Tokens per Second')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tokens_per_second_vs_batch_size.png")
        plt.close()
        

def main():
    """Main function to run the profiling."""
    print("Starting TinyLlama model profiling...")
    print(f"Using device: {DEVICE}")
    
    profiler = ModelProfiler(MODEL_NAME, DEVICE)
    
    print("\nStarting comprehensive profiling...")
    results = profiler.run_comprehensive_profile(
        num_runs=3,
        warmup_runs=1,
        output_dir="profiling_results"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "profiling_results"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = f"{output_dir}/profiling_results_{timestamp}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nProfiling results saved to: {csv_path}")
    
    profiler.plot_results(results, output_dir)
    print("Generated plots in the profiling_results directory.")
    
    print("\nProfiling Summary:")
    print(results[['batch_size', 'max_new_tokens', 
                    'avg_latency_ms', 'avg_tokens_per_second']].to_string())
        

if __name__ == "__main__":
    main()
