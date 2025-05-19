# Inference Optimization

Interview Type: Project

# Overview

In this exercise project, you will profile, benchmark, and optimize inference for a small language model under various scenarios. This challenge will test your ability to identify bottlenecks in LLM inference and implement effective optimizations while documenting your process and findings.

## Task Description

1. Select a small open-source model (e.g. TinyLlama)
2. Profile and benchmark performance across different scenarios, focus on the online serving scenario
3. Implement multiple performance optimizations
4. Measure and document the improvements
5. Present your findings

## Specific Requirements

### Model Selection

- Choose a small open-source LLM (TinyLlama recommended, but you may select another model if preferred)
- Document your rationale for any model-specific optimizations

### Runtime environment

A small model like TinyLlama should be able to be run on a Colab. Feel free to use any frameworks to improve performance over the naive `transformers` baseline, eg SGLang, TensorRT-LLM, vLLM, etc.

### Benchmarking Scenarios

Profile and benchmark the model's performance across various dimensions:

- Different batch sizes (1, 4, 8, 16, etc.)
- Various input sequence lengths
- Different output generation lengths
- Single vs. multiple requests

### Optimization Implementation

Implement any number of the following optimizations:

- Quantization techniques
- KV-cache optimizations
- Speculative decoding
- Attention mechanism optimizations
- Batching strategies
- Tensor parallelism
- Optimized CUDA kernels
- Other optimizations

### Evaluation and Documentation

- Document your entire process, including:
    - Initial benchmarking methodology and results
    - Identification of bottlenecks
    - Implementation details of each optimization
    - Performance improvements for each optimization
    - Tradeoffs between speed, memory, scalability and quality
- Depth of investigation is valued over breadth, ie feel free to go deep into benchmarking, profiling, specific optimization, ablation on tradeoffs, etc. if that’s interesting to you.
- Trying things that don’t work is also valued, document the things that were dead-ends.

## Deliverables

1. Code repository with clear structure and documentation
2. Writeup of your process and findings
3. A presentation (15-20 minutes) summarizing your work and being prepared to answer technical questions. Code/notebook walkthrough is sufficient for presentation. It will form the basis for a discussion that will be the main opportunity to go deep.

## Time Expectation

This challenge is the primary evaluation criteria for joining Baseten. It is designed to be completed in approximately 6-8 hours of focused work. We respect your time and don't expect you to spend more than a day on this project.

## Evaluation Criteria

- Technical understanding of LLM inference
- Thoroughness of benchmarking across different scenarios
- Quality, depth, and creativity of implemented optimizations
- Depth of analysis and clarity of documentation
- Communication skills and ability to explain technical concepts