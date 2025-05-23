#!/usr/bin/env python3
"""
Visualization script for vLLM concurrency profiling results.
Generates various plots to analyze the performance characteristics
under different concurrency levels, batch sizes, and token lengths.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the concurrency profile data."""
    df = pd.read_csv(csv_path)
    
    # Calculate additional metrics
    df['total_tokens'] = df['batch_size'] * df['max_tokens']
    df['requests_per_second'] = (df['batch_size'] * 1000) / df['avg_latency_ms']
    df['tokens_per_request'] = df['batch_size'] * df['max_tokens']
    
    return df

def plot_throughput_distribution(df: pd.DataFrame, output_dir: str):
    """Plot throughput distribution across different concurrency levels and batch sizes."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create a figure with subplots for each token length
    token_lengths = sorted(df['max_tokens'].unique())
    n_cols = min(2, len(token_lengths))
    n_rows = (len(token_lengths) + n_cols - 1) // n_cols
    
    # Adjust figure size based on number of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flat
    
    # Create a color palette
    batch_sizes = sorted(df['batch_size'].unique())
    palette = sns.color_palette("viridis", n_colors=len(batch_sizes))
    
    # Calculate bar width and positions
    n_batches = len(batch_sizes)
    bar_width = 0.8 / n_batches  # 80% of the space divided by number of batches
    
    for ax, token_len in zip(axes, token_lengths):
        # Filter data for this token length
        df_subset = df[df['max_tokens'] == token_len].copy()
        
        # Calculate mean and std for each group
        grouped = df_subset.groupby(['concurrency', 'batch_size'])['throughput_tps'].agg(['mean', 'std']).reset_index()
        
        # Plot each batch size
        for i, batch_size in enumerate(batch_sizes):
            batch_data = grouped[grouped['batch_size'] == batch_size]
            x_pos = np.arange(len(batch_data)) + i * bar_width - (n_batches - 1) * bar_width / 2
            
            # Plot bars with error bars
            ax.bar(
                x_pos,
                batch_data['mean'],
                width=bar_width * 0.9,  # 90% of the allocated width
                color=palette[i],
                label=f'Batch {batch_size}',
                alpha=0.8,
                edgecolor='white',
                linewidth=1,
                yerr=batch_data['std'],
                capsize=3,
                error_kw={'elinewidth': 1, 'capthick': 1}
            )
            
            # Add value labels on top of bars
            for j, (x, y) in enumerate(zip(x_pos, batch_data['mean'])):
                ax.text(
                    x, y + (y * 0.02),  # Slightly above the bar
                    f'{y:.0f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='black'
                )
        
        # Set x-ticks and labels
        x_ticks = np.arange(len(df_subset['concurrency'].unique()))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(sorted(df_subset['concurrency'].unique()))
        
        # Add grid and labels
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        ax.set_axisbelow(True)
        ax.set_xlabel('Concurrency Level', fontsize=12, labelpad=10)
        ax.set_ylabel('Throughput (tokens/s)', fontsize=12, labelpad=10)
        ax.set_title(f'Output Length = {token_len} tokens', fontsize=14, pad=15)
        
        # Add legend
        if i == 0:  # Only add legend once per subplot
            ax.legend(
                title='Batch Size',
                title_fontsize=11,
                fontsize=10,
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.
            )
    
    # Remove any empty subplots
    for i in range(len(token_lengths), len(axes)):
        fig.delaxes(axes[i])
    
    # Add a main title
    plt.suptitle(
        'Throughput vs Concurrency Level by Batch Size',
        fontsize=16,
        y=1.02
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(
        os.path.join(output_dir, 'throughput_distribution.png'),
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        transparent=False
    )
    plt.close()


def plot_latency_distribution(df: pd.DataFrame, output_dir: str):
    """Plot latency distribution across different concurrency levels and batch sizes."""
    # Set style
    sns.set_style("whitegrid")
    
    # Create a figure with subplots for each token length
    token_lengths = sorted(df['max_tokens'].unique())
    n_cols = min(2, len(token_lengths))
    n_rows = (len(token_lengths) + n_cols - 1) // n_cols
    
    # Adjust figure size based on number of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flat
    
    # Create a color palette
    batch_sizes = sorted(df['batch_size'].unique())
    palette = sns.color_palette("viridis", n_colors=len(batch_sizes))
    
    # Calculate bar width and positions
    n_batches = len(batch_sizes)
    bar_width = 0.8 / n_batches  # 80% of the space divided by number of batches
    
    for ax, token_len in zip(axes, token_lengths):
        # Filter data for this token length
        df_subset = df[df['max_tokens'] == token_len].copy()
        
        # Calculate mean and std for each group
        grouped = df_subset.groupby(['concurrency', 'batch_size'])['avg_latency_ms'].agg(['mean', 'std']).reset_index()
        
        # Plot each batch size
        for i, batch_size in enumerate(batch_sizes):
            batch_data = grouped[grouped['batch_size'] == batch_size]
            x_pos = np.arange(len(batch_data)) + i * bar_width - (n_batches - 1) * bar_width / 2
            
            # Plot bars with error bars
            ax.bar(
                x_pos,
                batch_data['mean'],
                width=bar_width * 0.9,  # 90% of the allocated width
                color=palette[i],
                label=f'Batch {batch_size}',
                alpha=0.8,
                edgecolor='white',
                linewidth=1,
                yerr=batch_data['std'],
                capsize=3,
                error_kw={'elinewidth': 1, 'capthick': 1}
            )
            
            # Add value labels on top of bars
            for j, (x, y) in enumerate(zip(x_pos, batch_data['mean'])):
                ax.text(
                    x, y + (y * 0.02),  # Slightly above the bar
                    f'{y:.1f}ms',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )
        
        # Set x-ticks and labels
        ax.set_xticks(np.arange(len(df_subset['concurrency'].unique())))
        ax.set_xticklabels([f'C{c}' for c in sorted(df_subset['concurrency'].unique())])
        
        # Set titles and labels
        ax.set_title(f'Output Length: {token_len}')
        ax.set_xlabel('Concurrency Level')
        ax.set_ylabel('Average Latency (ms)')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend to the first subplot only
        if token_len == token_lengths[0]:
            ax.legend(title='Batch Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove any empty subplots
    for i in range(len(token_lengths), n_rows * n_cols):
        fig.delaxes(axes[i])
    
    # Add a main title
    plt.suptitle(
        'Latency vs Concurrency Level by Batch Size',
        fontsize=16,
        y=1.02
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(
        os.path.join(output_dir, 'latency_distribution.png'),
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        transparent=False
    )
    plt.close()



def main(csv_path: str, output_dir: Optional[str] = None):
    """Main function to generate all plots."""
    # Set style
    sns.set_theme(style="whitegrid")
    plt.style.use('seaborn-v0_8')
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), 'concurrency_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    
    print("Generating plots...")
    
    # Generate individual plots
    plot_latency_distribution(df, output_dir)
    plot_throughput_distribution(df, output_dir)
    
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate concurrency analysis plots from vLLM profile data')
    parser.add_argument('--input', type=str, default='vllm_concurrent_profile.csv',
                       help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: <input_dir>/concurrency_plots)')
    
    args = parser.parse_args()
    
    main(args.input, args.output)
