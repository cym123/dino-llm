<p align="center">
<img width="400" src="/assets/logo.jpg">
</p>

面向生产环境的大语言模型高性能推理系统

---

##  项目简介
**dino-llm** 是一套工业级大模型推理系统。项目基于 HuggingFace Transformers 框架开发，解决推理延迟高、吞吐率低、硬件资源利用率不足等核心痛点。系统以动态批处理、KV 缓存优化、全链路可观测性为核心能力，可快速完成模型原型到线上稳定服务的落地交付。
##  核心功能

- 全链路可观测：接入普罗米修斯采集请求量、令牌吞吐、P50/P90/P95 延迟等指标，搭配格拉法纳可视化监控面板
- 分布式集群部署：采用路由节点 + 工作节点架构，内置多种负载均衡与节点自动健康检测
- 内置基准测试套件：配备 Locust 压测脚本、异步延迟吞吐测试、多协议性能对比工具
- 基数树缓存：复用请求间共用上下文 KV 缓存，大幅提升缓存命中率
- 分块预填充：拆分超长上下文预计算阶段，降低 GPU 显存峰值占用
- 重叠调度：CPU 调度逻辑与 GPU 计算并行执行，消除空等耗时
- 张量并行：无缝适配多卡模型分片推理，实现性能横向扩容
- 高速 CUDA 内核：深度融合闪注意力、FlashInfer 等优化算子，充分释放 GPU 算力

## Quick Start


```bash
# Create isolated virtual environment (Python 3.10+ required)
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
python -m dinollm --model "Qwen/Qwen3-0.6B"



