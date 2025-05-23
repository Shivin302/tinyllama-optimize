# TinyLlama Optimization Project

This project focuses on optimizing the inference performance of the TinyLlama model using various techniques such as quantization, KV-cache optimization, and batching strategies. The goal is to achieve maximum throughput and minimal latency while maintaining model quality.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- [UV package manager](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tinyllama-optimize.git
   cd tinyllama-optimize
   ```

2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. **Install the package in development mode**
   ```bash
   uv pip install -e .
   ```

4. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

## Project Structure

```
tinyllama-optimize/
├── tinyllama/
│   ├── profilers/         # Profiling scripts
│   ├── server/           # Server and client code
│   └── utils/            # Utility functions and plotting
├── examples/             # Example scripts
├── tests/                # Test files
├── requirements.txt      # Project dependencies
└── setup.py             # Package configuration
```

## Usage

### 1. Running the vLLM Server

To start the vLLM server with default settings:

```bash
python -m tinyllama.server.vllm_server
```

Available options:
```bash
python -m tinyllama.server.vllm_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 4096
```

### 2. Profiling the Model

#### Basic Profiling

Run the basic profiler to measure performance metrics:

```bash
python -m tinyllama.profilers.profile_vllm
```

#### Concurrency Profiling

To profile the server under different concurrency levels:

1. Start the vLLM server in a separate terminal
2. Run the concurrency profiler:
   ```bash
   python -m tinyllama.server.profile_vllm_server
   ```

This will generate plots in the `concurrency_plots` directory.

### 3. Generating Plots

To generate plots from existing profiling data:

```bash
python -m tinyllama.utils.plot_concurrency --input path/to/profile_data.csv
```

## Configuration

### Server Configuration

Modify server parameters in `tinyllama/server/vllm_server.py`:
- Model name/path
- Tensor parallelism
- Maximum sequence length
- KV cache settings

### Profiling Configuration

Adjust profiling parameters in `tinyllama/utils/profile.py`:
- Batch sizes
- Input/output sequence lengths
- Number of warmup and measurement runs

## Optimization Techniques

This project implements several optimization techniques:

1. **Quantization**
   - 8-bit and 4-bit quantization
   - FP8 KV cache

2. **Batching**
   - Dynamic batching
   - Continuous batching

3. **KV Cache Optimization**
   - PagedAttention
   - Memory-efficient caching

4. **Attention Optimization**
   - Flash Attention
   - Grouped-query attention

## Performance Metrics

Key metrics being tracked:
- Tokens per second (throughput)
- Latency (p50, p90, p99)
- Memory usage
- GPU utilization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Decrease sequence length
   - Enable quantization
   - Use gradient checkpointing

2. **Installation Issues**
   - Make sure CUDA is properly installed
   - Check Python version compatibility
   - Try reinstalling dependencies with `--no-cache-dir`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the base model
- [vLLM](https://github.com/vllm-project/vllm) for the efficient inference engine
- [Hugging Face](https://huggingface.co/) for model hosting and Transformers library
