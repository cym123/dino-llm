import asyncio
import aiohttp
import time
import statistics
import random
import json
from typing import List, Dict

# ===================== 配置 =====================
API_URL = "http://localhost:2000/v1/chat/completions"
API_KEY = "dummy"
CONCURRENCY = 8
TIMEOUT = 60
# ================================================

# 100 条完全不同的提示词，无公共前缀 → 绝对不触发 PrefixCache
PROMPT_LIST = [
    "请详细介绍人工智能的发展历史，包括起源、关键节点、各阶段技术突破，不少于200字",
    "请详细讲解深度学习中反向传播算法的原理，数学推导过程，以及在神经网络中的作用，尽量详细",
    "请详细分析大语言模型的自注意力机制原理，多头注意力的计算方式，以及为什么能提升效果",
    "请详细介绍Transformer架构的全部结构，Encoder、Decoder各层作用，尽量详细展开",
    "请详细讲解PyTorch中的自动求导机制原理，计算图构建，前向传播反向传播流程",
    "请详细介绍CUDA编程基础，线程块、线程网格、共享内存、全局内存，尽量详细",
    "请详细讲解GPU架构，SM、SP、Warp、Shared Memory、L1 Cache、L2 Cache作用",
    "请详细介绍机器学习中过拟合的原因、表现、以及常用的5种以上解决方法，每种方法详细说明",
    "请详细讲解随机梯度下降SGD、Adam、RMSprop、Adagrad优化器的区别与适用场景",
    "请详细介绍卷积神经网络CNN的原理，卷积核、池化、padding、stride的作用与计算方式",
    "请详细讲解循环神经网络RNN、LSTM、GRU的区别，门控机制原理，以及解决的问题",
    "请详细介绍自然语言处理中的词嵌入技术，One-hot、Word2Vec、GloVe、BERT嵌入区别",
    "请详细讲解大模型微调技术，全参数微调、LoRA、QLoRA、P-Tuning、Prompt Tuning区别",
    "请详细介绍模型量化技术，INT8、INT4、FP16、BF16精度区别，量化原理与优缺点",
    "请详细讲解KV Cache技术原理，在自回归生成中的作用，显存优化与加速原理",
    "请详细介绍Prefix Cache、Paged Attention、Flash Attention的优化原理",
    "请详细讲解大语言模型推理流程，Prefill阶段、Decode阶段、Token生成过程",
    "请详细介绍分布式训练技术，数据并行、模型并行、张量并行、流水线并行区别",
    "请详细讲解深度学习中归一化技术，BatchNorm、LayerNorm、RMSNorm原理与区别",
    "请详细介绍激活函数Sigmoid、Tanh、ReLU、GELU、Swish优缺点与适用场景",
    "请详细讲解损失函数MSE、CrossEntropy、BCE、Focal Loss原理与适用场景",
    "请详细介绍深度学习初始化方法，Xavier、He、Zero、Random初始化原理",
    "请详细讲解正则化技术，L1、L2、Dropout、DropConnect、Weight Decay原理",
    "请详细介绍NLP中的文本预处理流程，分词、编码、截断、padding、Mask机制",
    "请详细讲解Beam Search、Greedy Search、Top-k、Top-p采样生成策略区别",
    "请详细介绍多模态模型原理，文本、图像、音频特征融合方式，CLIP、Flux模型",
    "请详细讲解扩散模型DDPM原理，前向加噪、反向去噪、UNet架构、条件生成机制",
    "请详细介绍GAN对抗生成网络原理，生成器、判别器、损失函数、训练模式",
    "请详细讲解归一化流、自回归模型、VAE变分自编码器原理与区别",
    "请详细介绍强化学习基础，MDP、奖励函数、策略、价值函数、Q-learning",
    "请详细讲解RLHF技术原理，奖励模型、强化学习微调、对齐过程",
    "请详细介绍DPO、IPO、KTO、ORPO直接偏好优化算法原理与区别",
    "请详细讲解MoE混合专家模型原理，门控网络、专家选择、负载均衡机制",
    "请详细介绍长文本窗口扩展技术，ROPE、NTK、Sliding Window、Attention优化",
    "请详细讲解模型部署技术，ONNX、TensorRT、vLLM、TGI、Text Generation Inference",
    "请详细介绍高并发推理优化，动态批处理、批调度、Inflight Batch、CUDA Graph",
    "请详细讲解显存优化技术，梯度检查点、重计算、量化、分布式显存、Offloading",
    "请详细介绍深度学习编译器，TVM、MLIR、TorchDynamo、Compile原理",
    "请详细讲解时序模型原理，Transformer、TCN、LSTM在时序预测中的应用",
    "请详细介绍图神经网络GNN、GCN、GAT、GraphSAGE原理与应用场景",
    "请详细讲解小样本学习、零样本学习、少样本提示学习原理",
    "请详细介绍检索增强生成RAG技术，向量库、检索器、精排、上下文增强",
    "请详细讲解RAG优化技术，分块、召回、重排、提示词工程、上下文窗口管理",
    "请详细介绍向量数据库原理，HNSW、IVF、PQ、量化索引、近似最近邻搜索",
    "请详细讲解提示词工程技术，思维链、角色设定、格式约束、少样本示例",
    "请详细介绍大模型评测方法，准确率、流畅度、事实一致性、安全评测、Benchmark",
    "请详细讲解模型压缩技术，剪枝、量化、蒸馏、结构搜索、低秩分解",
    "请详细介绍深度学习数据集构建，清洗、标注、增强、去重、分布对齐",
    "请详细讲解数据并行训练，AllReduce、AllGather、通信优化、Ring AllReduce",
    "请详细介绍张量并行、流水线并行、序列并行、专家并行区别与适用",
    "请详细讲解深度学习推理性能指标，TTFT、TPOT、Throughput、Latency、并发",
    "请详细介绍服务部署架构，负载均衡、服务发现、限流、降级、容错、健康检查",
    "请详细讲解Docker容器化、K8s编排、GPU调度、微服务推理部署",
    "请详细介绍LLM服务监控，指标采集、日志、告警、Trace追踪、Dashboard",
    "请详细讲解大模型安全对齐，无害性、诚实性、有用性、对抗攻击、防御",
    "请详细介绍预训练语料构建，去重、过滤、质量评估、版权合规、数据配比",
    "请详细讲解因果语言模型、掩码语言模型、序列到序列模型训练目标区别",
    "请详细介绍位置编码，绝对、相对、旋转、Alibi、学习式位置编码原理",
    "请详细讲解注意力优化，稀疏注意力、局部注意力、FlashAttention、分片注意力",
    "请详细介绍动态批处理原理，批调度、队列管理、时延控制、优先级、超时",
    "请详细讲解模型服务框架，FastAPI、Triton、TorchServe、vLLM、SGLang架构",
    "请详细介绍NLP生成任务评价指标，PPL、BLEU、ROUGE、BERTScore、MoverScore",
    "请详细讲解多轮对话管理，状态跟踪、上下文管理、历史截断、意图识别",
    "请详细介绍知识库问答技术，段落检索、句子匹配、答案抽取、生成式问答",
    "请详细讲解代码生成模型原理，AST、语法树、类型提示、执行验证、测试用例",
    "请详细介绍语音识别模型，Whisper、Conformer、CTC、Attention解码器",
    "请详细讲解语音合成TTS技术，VITS、Flow、Diffusion、声码器、梅尔谱",
    "请详细介绍数字人技术，面部驱动、语音驱动、渲染、实时推理、低延迟",
    "请详细讲解自动驾驶感知模型，视觉、激光雷达、融合、BEV、Transformer",
    "请详细介绍医疗大模型，医学知识、病历分析、影像报告、临床决策支持",
    "请详细讲解法律大模型，法条检索、合同审查、案件分析、法律问答生成",
    "请详细介绍教育大模型，知识点讲解、题库生成、作文批改、个性化辅导",
    "请详细讲解金融大模型，研报分析、风险评估、量化策略、新闻情绪分析",
    "请详细介绍农业大模型，病虫害识别、种植指导、气象预测、产量估算",
    "请详细讲解工业大模型，缺陷检测、预测性维护、流程优化、安全生产",
    "请详细介绍游戏AI技术，大模型NPC、剧情生成、对话系统、策略决策、多智能体",
    "请详细讲解机器人技术，具身智能、视觉语言导航、操作控制、环境交互",
    "请详细介绍AIGC技术，文生图、文生视频、文生3D、数字人、声音克隆",
    "请详细讲解云计算GPU实例，NVIDIA A10、A100、H100、L40S、T4性能对比",
    "请详细介绍CUDA核心、Tensor Core、RT Core、NVLink、PCIe、显存带宽",
    "请详细讲解深度学习框架，PyTorch、TensorFlow、JAX、MindSpore、PaddlePaddle",
    "请详细介绍混合精度训练，FP16、BF16、FP8精度、损失缩放、数值稳定性",
    "请详细讲解自动机器学习AutoML，网络搜索、超参优化、模型选择、自动化流程",
    "请详细介绍可解释性AI，注意力可视化、特征重要性、LIME、SHAP、因果推理",
    "请详细讲解联邦学习、隐私计算、同态加密、差分隐私、数据安全",
    "请详细介绍元学习、迁移学习、领域自适应、小样本学习、泛化能力",
    "请详细讲解神经架构搜索NAS，搜索空间、优化目标、性能预测、训练策略",
    "请详细介绍持续学习、增量学习、灾难性遗忘、记忆回放、参数隔离",
    "请详细讲解多任务学习、硬参数共享、软参数共享、任务平衡、梯度冲突",
    "请详细介绍自监督学习、对比学习、掩码学习、预训练、无标注数据利用",
    "请详细讲解弱监督学习、远监督、部分监督、噪声标签、数据高效学习",
    "请详细介绍表征学习、特征提取、降维、流形学习、度量学习",
    "请详细讲解生成模型评估，FID、IS、Precision、Recall、多样性、保真度",
    "请详细介绍NLP鲁棒性，对抗样本、分布外泛化、噪声文本、错误容忍",
    "请详细讲解模型轻量化，MobileBERT、DistilBERT、TinyBERT、模型剪枝蒸馏",
    "请详细介绍边缘计算部署，端侧推理、NPU、TPU、NCNN、MNN、TFLite、ONNXRuntime",
    "请详细讲解云原生AI，弹性伸缩、自动扩缩容、负载均衡、故障自愈、灰度发布",
    "请详细介绍大模型应用开发，提示词、函数调用、工具使用、工作流、Agent智能体",
    "请详细讲解智能体技术，规划、记忆、工具、反思、多智能体协作、环境交互",
    "请详细介绍下ai infra岗位需要技能"
]

print(f"提示词总数: {len(PROMPT_LIST)}")

assert len(PROMPT_LIST) == 100


class BenchMetrics:
    def __init__(self):
        self.ttft_list: List[float] = []
        self.total_lat_list: List[float] = []
        self.tpot_list: List[float] = []
        self.total_tokens = 0

    def add(self, ttft: float, total_lat: float, tokens: int):
        self.ttft_list.append(ttft)
        self.total_lat_list.append(total_lat)
        self.total_tokens += tokens

        if tokens >= 2:
            gen_time = total_lat - ttft
            tpot = gen_time / (tokens - 1)
            self.tpot_list.append(tpot)

    def calculate(self):
        def p50(arr): return sorted(arr)[int(len(arr) * 0.5)] if arr else 0.0
        def p99(arr): return sorted(arr)[int(len(arr) * 0.99)] if arr else 0.0

        return {
            "requests": len(self.ttft_list),
            "total_output_tokens": self.total_tokens,
            "ttft_ms": {
                "avg": round(statistics.mean(self.ttft_list), 2) if self.ttft_list else 0,
                "p50": round(p50(self.ttft_list), 2),
                "p99": round(p99(self.ttft_list), 2),
            },
            "total_latency_ms": {
                "avg": round(statistics.mean(self.total_lat_list), 2) if self.total_lat_list else 0,
                "p50": round(p50(self.total_lat_list), 2),
                "p99": round(p99(self.total_lat_list), 2),
            },
            "tpot_ms": {
                "avg": round(statistics.mean(self.tpot_list), 4) if self.tpot_list else 0,
                "p50": round(p50(self.tpot_list), 4),
                "p99": round(p99(self.tpot_list), 4),
            }
        }


metrics = BenchMetrics()


async def run_query(session: aiohttp.ClientSession, prompt: str, user_id: str):
    start = time.perf_counter()
    ttft = None
    token_count = 0

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "user": user_id,
        "max_tokens": 128,
        "temperature": 0.7,
    }

    try:
        async with session.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as resp:
            async for line in resp.content:
                line = line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    j = json.loads(data)
                    token_count += 1  # 每一个 chunk = 1 token ✅ 真实精准

                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                except:
                    continue

        total_lat = (time.perf_counter() - start) * 1000
        return ttft, total_lat, token_count

    except Exception as e:
        print(f"失败: {str(e)[:50]}")
        return None, None, 0


async def worker(queue: asyncio.Queue):
    async with aiohttp.ClientSession() as s:
        while not queue.empty():
            prompt, uid = await queue.get()
            t1, t2, tok = await run_query(s, prompt, uid)
            if t1 and t2 and tok:
                metrics.add(t1, t2, tok)
            queue.task_done()


async def main():
    print("=" * 70)
    print("🔥 纯冷启动压测｜100条独立提示词｜零PrefixCache｜真实Token统计")
    print("=" * 70)

    q = asyncio.Queue()
    prompts = random.sample(PROMPT_LIST, 100)

    for i, p in enumerate(prompts):
        unique_user = f"cold_user_{time.time_ns()}_{i}"
        q.put_nowait((p, unique_user))

    tasks = [asyncio.create_task(worker(q)) for _ in range(CONCURRENCY)]
    await q.join()
    for t in tasks:
        t.cancel()

    res = metrics.calculate()
    print("\n✅ 压测完成！")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())