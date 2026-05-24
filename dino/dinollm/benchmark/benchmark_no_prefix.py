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
CONCURRENCY = 256
TIMEOUT = 60
TEST_REQUESTS = 256
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
    "请详细介绍下ai infra岗位需要技能",
    "智能体规划拆解复杂任务层级，按步骤推进目标有序落地执行",
    "长短记忆分类存储交互信息，调取历史数据支撑实时决策判断",
    "智能体灵活调用各类工具接口，完成计算检索查询实操类任务",
    "执行完毕开启自我反思复盘，修正行为缺陷优化后续处事逻辑",
    "多智能体分工配合信息互通，协作联动共同处理复合型难题",
    "实时感知外部环境动态变化，自主调整行动策略适配场景需求",
    "优先级规划梳理任务顺序，精简执行流程降低资源无谓消耗",
    "短时记忆留存临时对话内容，保障交互顺畅应答节奏不中断",
    "拓展工具体系拓宽能力范围，独立应对不同类型实际业务场景",
    "定期反思过往决策得失，沉淀经验稳步提升智能判断水准",
    "全局统筹规划行动路线，规避执行风险保障任务稳步推进",
    "长效记忆沉淀行为数据，依托过往经历优化当下选择判断",
    "按需匹配专业工具模块，高效处理图文数据与运算类工作",
    "深度反思行动疏漏问题，迭代算法模型强化智能响应能力",
    "多智能体信息共享互通，划分职责协同完成大规模作业",
    "敏锐捕捉环境细微变动，即时调整行为模式适配外界变化",
    "动态规划适配突发状况，灵活变通方案保障任务正常推进",
    "记忆检索快速调取资料，依托存量信息缩短决策耗时周期",
    "跨界工具组合搭配使用，一站式解决多元复杂现实问题",
    "复盘全过程总结优缺点，持续迭代升级智能体核心性能",
    "群组智能体协同调度，相互配合弥补个体能力存在短板",
    "沉浸式感知周遭环境，依据场景特征制定适配行动方案",
    "精细化拆分规划目标，细化执行步骤提升整体作业效率",
    "记忆分层管理存取数据，精准筛选有效信息辅助逻辑推理",
    "智能体自主发起工具调用，自主研判场景选用适配功能",
    "对照执行结果反向反思，找准不足优化后续行动策略",
    "跨智能体协商沟通协作，统一目标步调合力攻克难点",
    "主动探测环境状态参数，依据实时数据调整运行模式",
    "远期规划锁定最终方向，分步拆解小目标稳步趋近结果",
    "关联记忆串联相关内容，整合碎片化信息形成完整依据",
    "专用工具赋能专项任务，凭借功能优势高效处理对应事务",
    "阶段性反思行为成效，及时调整思路避免偏离既定方向",
    "多主体协同分配工作，各司其职衔接顺畅提升整体效能",
    "持续监测环境演变趋势，预判变化提前做好应对预备举措",
    "智能规划规避无效动作，合理分配算力资源提升运转效率",
    "记忆归档留存交互轨迹，追溯过往过程分析决策合理性",
    "集成多样工具拓展本领，全方位满足各类场景使用需求",
    "深度反思决策思维漏洞，优化思考模式减少判断失误情况",
    "联盟智能体协同联动，互通资源优势互补化解棘手问题",
    "动态适配环境时空变化，顺势调整行动方式契合现场条件",
    "逆向规划倒推执行路径，反向梳理环节保障任务闭环落地",
    "联想记忆挖掘信息关联，挖掘隐藏线索助力问题分析研判",
    "智能调度工具组合功能，搭配操作完成综合性复杂事务",
    "事后反思梳理成败缘由，汲取经验教训优化后续处置方式",
    "分布式多智能体协作，区域分工联动协作完成全域任务",
    "立体感知多维环境信息，综合研判现状制定合理行动方案",
    "弹性规划适配任务难度，灵活调整节奏适配不同作业体量",
    "记忆压缩精简核心内容，剔除冗余信息加快检索响应速度",
    "智能识别场景匹配工具，精准选用功能适配当下处理诉求",
    "常态化反思迭代能力，不断修正行为趋近最优执行效果",
    "异构智能体配合协作，差异化能力互补覆盖全业务流程",
    "实时反馈环境交互结果，依据反馈内容优化自身运行状态",
    "策略规划博弈应对场景，权衡利弊选择最优行动实施方案",
    "记忆溯源还原交互细节，还原过程原貌辅助问题复盘总结",
    "自动化调用内置工具，无需人工干预自主完成常规作业",
    "批判性反思固有思维，打破局限思路提升决策创新程度",
    "集群智能体统一调度，统一指令步调一致协同推进项目",
    "主动交互环境获取信息，采集基础数据支撑后续分析运算",
    "模块化规划拆分大任务，独立模块作业降低执行出错概率",
    "记忆加密保护交互数据，安全存储信息规避内容泄露风险",
    "工具权限分级管控使用，规范调用流程保障操作安全稳定",
    "对比反思不同执行方案，择优选用模式提升任务完成质量",
    "层级化多智能体管理，上下级联动统筹推进整体工作进度",
    "环境边界感知规避风险，远离危险区域保障智能体运行安全",
    "预判式规划提前布局，结合趋势动向提前安排后续工作",
    "记忆标签分类规整数据，快速定位所需信息缩短查找时间",
    "轻量化工具快速调用，简易操作高效处理日常基础事务",
    "自我反思优化交互话术，调整表达逻辑提升沟通流畅程度",
    "对等智能体平等协作，互相商议探讨共同敲定处置方案",
    "环境要素拆解分析研判，抓住核心条件制定对应应对举措",
    "约束规划划定行动边界，恪守规则范围合规开展各项作业",
    "记忆遗忘机制清理废数据，释放存储空间保障运转流畅度",
    "重型工具承载复杂运算，依靠强大算力处理高阶分析任务",
    "全局反思统筹整体流程，梳理衔接卡点优化整体运转链路",
    "跨领域智能体跨界协作，融合多类技术解决跨界疑难问题",
    "环境态势评估预判走向，提前预估变化提早筹备应对手段",
    "迭代规划逐步优化方案，依照执行效果持续微调行动步骤",
    "记忆联动跨时段调取内容，串联前后信息形成完整逻辑链",
    "工具自定义适配专属需求，按需调整参数贴合实际使用场景",
    "细节反思排查微小疏漏，补齐细节短板完善整体执行效果",
    "分布式决策多智能体商议，汇集多方意见敲定最终执行方案",
    "环境干扰过滤剔除无效信息，保留有效内容支撑精准判断",
    "协同规划多方商定路线，凝聚共识确定统一任务推进方向",
    "记忆快照留存关键节点，定格重要时刻方便后续复盘查阅",
    "工具协同联动配合运作，多项功能同步发力加速任务办结",
    "归因反思剖析问题根源，直击核心症结从本源解决各类问题",
    "动态组网多智能体集群，灵活组合队伍适配不同作业规模",
    "环境资源勘测采集素材，整合可用资源最大化发挥利用价值",
    "风险规划预判潜在隐患，提前制定预案规避各类突发意外",
    "记忆迁移复用过往经验，借鉴相似场景做法简化当下处理",
    "智能工具自主优化参数，自适应调整配置适配运行工况",
    "成长反思总结进阶经验，积累阅历持续拔高智能综合水准",
    "区域协同智能体分片作业，分区管控高效覆盖大范围场景",
    "环境氛围感知调整交互语气，贴合场景氛围开展顺畅沟通",
    "目标规划锚定核心诉求，舍弃无关内容聚焦关键任务要点",
    "记忆备份保障数据安全，多重留存方式防止重要信息丢失",
    "便携工具便捷随时调用，随时随地响应临时突发处理需求",
    "偏差反思修正认知误区，扭转错误判断回归客观事实本身",
    "任务委派多智能体分工履职，专人专项负责保障事务高效推进",
    "环境地形适配调整移动轨迹，贴合空间形态平稳完成位移动作",
    "资源规划合理调配可用物资，科学分配物料保障作业持续运转",
    "记忆融合整合多源信息，汇总各方数据形成全面参考依据",
    "开源工具拓展功能储备，吸纳多元能力丰富自身处理范畴",
    "复盘反思梳理全程得失，总结优势不足推动能力稳步进阶",
    "消息互通多智能体实时传讯，即时同步动态保持信息步调统一",
    "环境温度气压等参数感知，依据物理条件调整自身运行状态",
    "时序规划依照时间节点推进，准时完成各阶段既定工作目标",
    "记忆筛选甄别有效讯息，剔除虚假内容筑牢判断真实基础",
    "仿真工具模拟场景推演，预判执行效果提前优化行动方案",
    "认知反思刷新固有认知，接纳全新思路拓宽思考判断眼界",
    "等级协作智能体层级配合，逐级传达指令逐层落实工作内容",
    "环境声响影像多模态感知，多维度捕捉信息全面认知外界情况",
    "成本规划严控资源消耗额度，节约损耗实现高效低碳运行模式",
    "记忆索引搭建检索体系，建立查询目录极速定位目标信息",
    "办公工具助力文案整理编撰，规范格式内容完成文书类工作",
    "行为反思规范自身举动，优化行事方式塑造稳定作业风格",
    "联盟组网多智能体抱团协作，凝聚集体力量攻克超高难度任务",
    "环境气流光影细微变化捕捉，敏锐察觉异动及时做出对应反应",
    "预案规划备好多种处置方案，不同状况切换对应策略灵活应对",
    "记忆时序排序梳理先后内容，依照时间脉络理清事件发展过程",
    "测绘工具勘测空间位置数据，精准定位坐标掌握环境分布情况",
    "价值反思衡量行动实际意义，舍弃无用行为聚焦高价值事务",
    "互助协作智能体彼此帮扶，弥补个体短板共同提升作业成效",
    "环境人群氛围感知适配交流，贴合群体状态调整沟通表达模式",
    "路径规划测算最优行进线路，缩减行进距离节省运转时间成本",
    "记忆快照对比前后信息差异，对照变化内容分析事态演变走向",
    "统计工具核算汇总各类数据，梳理数值规律提炼有效分析结论",
    "决策反思审视选择合理程度，复盘抉择过程优化后续判断思维",
    "跨境域智能体信息互通共享，打破领域壁垒融合多元技术思路",
    "环境路况阻碍识别规避绕行，避开障碍物保障行进过程安全顺畅",
    "产能规划匹配任务处理体量，调整作业速率适配当下工作负荷",
    "记忆脱敏处理隐私类信息，合规加工内容规避隐私泄露问题",
    "编程工具编写调试功能代码，搭建程序模块实现定制化业务功能",
    "效果反思对照预期查验成果，比对标准差距针对性优化改进",
    "梯队协作智能体前后衔接配合，首尾呼应连贯完成整套作业流程",
    "环境昼夜明暗光线变化适应，调整视觉感知模式适配光照条件",
    "弹性目标规划预留调整空间，应对变数情况灵活改动任务指标",
    "记忆聚类归纳同类信息内容，整合相似条目简化信息管理流程",
    "测绘检测工具核查实体状态，核验物件参数判断实际完好程度",
    "思维反思重构思考逻辑框架，理顺推理脉络提升分析判断精度",
    "联动组网多智能体跨区配合，跨越空间距离协同处置关联事务",
    "环境酸碱湿度理化特性感知，适配环境特质保障设备稳定运作",
    "极简规划删减冗余执行环节，压缩流程步骤提升整体运转速率",
    "记忆时效判定信息有效期限，过期内容清理保证信息参考价值",
    "影音工具剪辑处理音视频素材，修整画面音效产出合规成品内容",
    "协作反思梳理配合衔接问题，优化联动方式提升团队协作默契",
    "子母智能体从属配合开展工作，上级统筹把控下级落地执行事务",
    "环境人群疏密状态实时感知，依据人流密度调整交互行动尺度",
    "博弈规划权衡多方利益关系，平衡各方诉求拟定折中处置方案",
    "记忆拼接整合碎片化资讯，拼凑完整全貌还原事件真实原貌",
    "计算工具高速演算复杂公式，精准得出数值结果支撑数据分析",
    "成长反思对标行业优秀水准，找寻自身差距朝着标杆看齐进步",
    "轮值协作智能体交替值守作业，轮流履职保障服务不间断运行",
    "环境季节气候更迭状态适配，顺应气候特征调整日常行动模式",
    "闭环规划首尾衔接形成完整链路，全程把控流程保障任务圆满收官",
    "记忆溯源追踪信息来源出处，核实内容真伪确保参考资料可靠"
]

print(f"提示词总数: {len(PROMPT_LIST)}")

assert len(PROMPT_LIST) == 256


class BenchMetrics:
    def __init__(self):
        self.ttft_list: List[float] = []
        self.total_lat_list: List[float] = []
        self.tpot_list: List[float] = []
        self.total_tokens = 0
        self.total_requests = 0

    def add(self, ttft: float, total_lat: float, tokens: int):
        self.ttft_list.append(ttft)
        self.total_lat_list.append(total_lat)
        self.total_tokens += tokens
        self.total_requests += 1

        if tokens >= 2:
            gen_time = total_lat - ttft
            tpot = gen_time / (tokens - 1)
            self.tpot_list.append(tpot)

    def calculate(self, total_time: float):
        def p50(arr): return sorted(arr)[int(len(arr) * 0.5)] if arr else 0.0
        def p90(arr): return sorted(arr)[int(len(arr) * 0.9)] if arr else 0.0
        def p95(arr): return sorted(arr)[int(len(arr) * 0.95)] if arr else 0.0
        def p99(arr): return sorted(arr)[int(len(arr) * 0.99)] if arr else 0.0

        qps = self.total_requests / total_time if total_time > 0 else 0
        token_throughput = self.total_tokens / total_time if total_time > 0 else 0

        return {
            "total_time_sec": round(total_time, 2),
            "requests": self.total_requests,
            "total_output_tokens": self.total_tokens,
            "qps": round(qps, 2),
            "token_throughput": round(token_throughput, 2),
            "ttft_ms": {
                "avg": round(statistics.mean(self.ttft_list), 2) if self.ttft_list else 0,
                "p50": round(p50(self.ttft_list), 2),
                "p90": round(p90(self.ttft_list), 2),
                "p95": round(p95(self.ttft_list), 2),
                "p99": round(p99(self.ttft_list), 2),
            },
            "total_latency_ms": {
                "avg": round(statistics.mean(self.total_lat_list), 2) if self.total_lat_list else 0,
                "p50": round(p50(self.total_lat_list), 2),
                "p90": round(p90(self.total_lat_list), 2),
                "p95": round(p95(self.total_lat_list), 2),
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
    max_tokens = random.randint(100, 1024)
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "user": user_id,
        "max_tokens": max_tokens,
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
                    json.loads(data)
                    token_count += 1
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                except:
                    continue

        total_lat = (time.perf_counter() - start) * 1000
        return ttft, total_lat, token_count

    except Exception as e:
        print(f"请求失败: {str(e)[:50]}")
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
    print("=" * 80)
    print("🔥 LLM 推理基准测试（无缓存冷启动｜真实Token统计｜专业Benchmark）")
    print("=" * 80)

    q = asyncio.Queue()
    
    expanded_prompts = []
    selected_prompts = random.sample(PROMPT_LIST, TEST_REQUESTS)
    
    for s in selected_prompts:
        # 随机生成 3 到 30 之间的整数（包含 3 和 30）
        repeat_times = random.randint(2, 29)
        # 让内容变长，但还是 1 条！
        long_prompt = s + " " + "全方位剖析技术原理细节，多角度对比优劣差异，完整阐述落地应用要点" * repeat_times
        expanded_prompts.append(long_prompt)
        
    print("\n📏 每条提示词最终长度（字符数）：")
    for idx, text in enumerate(expanded_prompts):
        print(f"第 {idx+1} 条长度：{len(text)}")
    

    for i, p in enumerate(expanded_prompts):
        unique_user =f"bench_user_{time.time_ns()}_{i}"
        q.put_nowait((p, unique_user))

    start_all = time.perf_counter()
    tasks = [asyncio.create_task(worker(q)) for _ in range(CONCURRENCY)]
    await q.join()
    total_time = time.perf_counter() - start_all

    for t in tasks:
        t.cancel()

    res = metrics.calculate(total_time)
    print("\n✅ 测试完成！\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())