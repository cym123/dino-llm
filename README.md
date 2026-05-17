<p align="center">
<img width="400" src="/assets/logo.jpg">
</p>

A **production-oriented high-performance** inference system for Large Language Models.

---

## 📌 Project Intro
**dino-llm** is a production-grade LLM inference system focused on practical ML system engineering. Built upon HuggingFace Transformers, it solves core pain points including high inference latency, low throughput and low hardware resource utilization.
Centered on dynamic batching, KV cache optimization and full observability, it supports full-containerized deployment, enabling rapid delivery from prototype to stable online LLM serving.

## ✨ Core Features

### System Capabilities
- **Full-link Observability**: Integrate Prometheus for metrics collection (request count, token throughput, P50/P90/P95 latency histogram), equipped with Grafana real-time monitoring dashboard.
- **Dual Protocol Support**: Native support for FastAPI HTTP and gRPC service interfaces, convenient for performance comparison and business access.
- **Distributed Cluster Deployment**: Multi-node cluster architecture with router node + stateless worker nodes, built-in polling load balance and automatic health check.
- **Built-in Benchmark Suite**: Provide Locust pressure test scripts, async latency & throughput testing tools, cross-protocol performance comparison tools.
- **Linux Resource Tuning**: Bundled CPU binding, process monitoring and performance profiling scripts to maximize server resource efficiency.

### Advanced Inference Optimizations
- **Radix Cache**: Reuse shared prefix KV cache among different requests to boost cache hit rate.
- **Chunked Prefill**: Split long context prefill phase to cut down peak GPU memory consumption.
- **Overlap Scheduling**: Overlap CPU scheduling logic with GPU computing execution to eliminate idle waiting delay.
- **Tensor Parallelism**: Seamless multi-GPU inference splitting to achieve horizontal performance scaling.
- **High-Speed CUDA Kernels**: Deeply integrate FlashAttention, FlashInfer and other optimized kernels to fully release GPU computing power.

## 🚀 Quick Start

### 1. Environment Setup
We recommend using `uv` for fast and reliable dependency management, compatible with conda environment.

```bash
# Create isolated virtual environment (Python 3.10+ required)
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .


