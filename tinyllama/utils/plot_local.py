import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import argparse


def plot_results_input_constant(df: pd.DataFrame, output_dir: str = "profiling_results"):
    """Plot profiling results with support for different output lengths."""
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
    


@staticmethod
def plot_results_output_constant(df: pd.DataFrame, output_dir: str = "profiling_results"):
    """Plot profiling results with support for different input lengths."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot latency vs batch size for different input lengths and output lengths
    plt.figure(figsize=(14, 8))
    
    # Create a grid of subplots: one row per input length
    fig, axes = plt.subplots(len(df['max_new_tokens'].unique()), 1, 
                            figsize=(14, 6 * len(df['max_new_tokens'].unique())))
    if len(df['max_new_tokens'].unique()) == 1:
        axes = [axes]  # Ensure axes is always a list
        
    for ax, (output_len, group) in zip(axes, df.groupby('max_new_tokens')):
        for max_tokens in sorted(group['input_length'].unique()):
            subset = group[group['input_length'] == max_tokens]
            ax.errorbar(
                subset['batch_size'],
                subset['avg_latency_ms'],
                yerr=subset['std_latency_ms'],
                label=f'Input={max_tokens} tokens',
                marker='o',
                capsize=5
            )
        
        ax.set_title(f'Output Length: {output_len} tokens')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_vs_batch_size2.png")
    plt.close()
    
    # Plot tokens per second vs batch size
    fig, axes = plt.subplots(len(df['max_new_tokens'].unique()), 1, 
                            figsize=(14, 6 * len(df['max_new_tokens'].unique())))
    if len(df['max_new_tokens'].unique()) == 1:
        axes = [axes]  # Ensure axes is always a list
        
    for ax, (output_len, group) in zip(axes, df.groupby('max_new_tokens')):
        for max_tokens in sorted(group['input_length'].unique()):
            subset = group[group['input_length'] == max_tokens]
            ax.errorbar(
                subset['batch_size'],
                subset['avg_tokens_per_second'],
                yerr=subset['std_tokens_per_second'],
                label=f'Input={max_tokens} tokens',
                marker='o',
                capsize=5
            )
        
        ax.set_title(f'Output Length: {output_len} tokens')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Tokens per Second')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tokens_per_second_vs_batch_size2.png")
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot profiling results.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing profiling results.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots.")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    if args.output_dir is None:
        output_dir = os.path.dirname(args.csv_path)
        output_dir = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_results_input_constant(df, output_dir)
    plot_results_output_constant(df, output_dir)