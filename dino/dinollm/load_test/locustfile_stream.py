from locust import HttpUser, task, between, events
import random
import json
import time
import threading

# ==================== 全局配置 ====================
# 生成的最大token长度
MAX_TOKENS = 256
# 生成温度（随机性）
TEMPERATURE = 0.7
# 用户思考时间（请求间隔）
MIN_WAIT = 0.5
MAX_WAIT = 1.0
# OpenAI 兼容接口地址
API_PATH = "/v1/chat/completions"

# 测试用的提示词池，随机抽取
MSG_POOL = [
    [{"role": "user", "content": "Explain quantum computing simply."}],
    [{"role": "user", "content": "Write a short robot story."}],
    [{"role": "user", "content": "How to optimize LLM inference?"}],
    [{"role": "user", "content": "Future development of artificial intelligence."}],
    [{"role": "user", "content": "Introduce machine learning basics."}],
]

# ==================== 全局指标收集 ====================
class PerfMetrics:
    def __init__(self):
        # 线程锁，保证多用户压测时数据不乱
        self.lock = threading.Lock()
        # 总请求数
        self.req_count = 0
        # 总生成token数
        self.total_tokens = 0

        # 首包时间列表（TTFT）
        self.ttft_list = []
        # 单token耗时列表（TPOT）
        self.tpot_list = []
        # 总延迟列表
        self.full_latency_list = []
        # 压测开始时间
        self.start_time = time.time()

    # 添加一次请求的指标数据
    def add_data(self, ttft: float, tpot: float, full_latency: float, token_cnt: int):
        with self.lock:
            self.req_count += 1
            self.total_tokens += token_cnt
            # 全部转毫秒保存
            self.ttft_list.append(ttft * 1000)
            self.tpot_list.append(tpot * 1000)
            self.full_latency_list.append(full_latency * 1000)

    # 计算百分位：P50/P90/P95/P99
    def percentile(self, data, p):
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[idx]

    # 统计所有指标
    def get_stats(self):
        if self.req_count == 0:
            return None
        
        # 压测总耗时
        total_time = time.time() - self.start_time
        # QPS = 总请求数 / 总时间
        qps = self.req_count / total_time if total_time > 0 else 0
        # Token吞吐 = 总token / 总时间
        throughput = self.total_tokens / total_time if total_time > 0 else 0

        return {
            "req": self.req_count,
            "tokens": self.total_tokens,
            "time_s": round(total_time, 2),
            "qps": round(qps, 2),
            "throughput": round(throughput, 2),

            # 平均时间（毫秒）
            "avg_ttft": round(sum(self.ttft_list)/len(self.ttft_list), 2),
            "avg_tpot": round(sum(self.tpot_list)/len(self.tpot_list), 2),
            "avg_lat": round(sum(self.full_latency_list)/len(self.full_latency_list), 2),

            # TTFT 百分位（毫秒）
            "ttft_p50": round(self.percentile(self.ttft_list, 50), 2),
            "ttft_p90": round(self.percentile(self.ttft_list, 90), 2),
            "ttft_p95": round(self.percentile(self.ttft_list, 95), 2),
            "ttft_p99": round(self.percentile(self.ttft_list, 99), 2),

            # 总延迟百分位（毫秒）
            "lat_p50": round(self.percentile(self.full_latency_list, 50), 2),
            "lat_p90": round(self.percentile(self.full_latency_list, 90), 2),
            "lat_p95": round(self.percentile(self.full_latency_list, 95), 2),
            "lat_p99": round(self.percentile(self.full_latency_list, 99), 2),
        }

# 全局指标实例
metrics = PerfMetrics()

# ==================== 压测用户类 ====================
class OpenAIStreamUser(HttpUser):
    # 用户请求间隔：0.5~2秒随机
    wait_time = between(MIN_WAIT, MAX_WAIT)
    # OpenAI API 请求头
    headers = {"Content-Type": "application/json", "Authorization": "Bearer none"}

    @task
    def chat_stream(self):
        # 随机选一个提示词
        msg = random.choice(MSG_POOL)
        # 随机生成长度 64~256 token
        max_tokens = random.randint(64, MAX_TOKENS)
        
        # OpenAI 标准请求体
        payload = {
            "model": "auto",
            "messages": msg,
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": TEMPERATURE
        }

        # 请求开始时间
        start = time.perf_counter()
        # 首token时间
        first_token = None
        # 生成的token计数
        tokens = 0

        # 发送流式POST请求
        with self.client.post(
            API_PATH,
            headers=self.headers,
            json=payload,
            stream=True,
            catch_response=True,
            timeout=120
        ) as resp:
            # HTTP状态码不是200直接标记失败
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return

            try:
                # 逐行读取流式返回
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # 第一次收到数据 = 首token时间（TTFT）
                    if first_token is None:
                        first_token = time.perf_counter()
                    # 统计有效token
                    if line.startswith(b"data: ") and b"[DONE]" not in line:
                        tokens += 1

                # 计算耗时（秒）
                ttft = first_token - start if first_token else 0
                full_lat = time.perf_counter() - start
                tpot = (full_lat - ttft) / max(tokens, 1)

                # 存入指标
                metrics.add_data(ttft, tpot, full_lat, tokens)
                resp.success()

            except Exception as e:
                resp.failure(f"Error: {str(e)}")

# ==================== 测试结束输出报告 ====================
@events.test_stop.add_listener
def on_stop(**kwargs):
    s = metrics.get_stats()
    if not s:
        print("\n❌ 无有效请求")
        return

    print("\n" + "="*70)
    print("               LLM 流式压测 完整性能报告 (毫秒版)")
    print("="*70)
    print(f"压测时长：{s['time_s']} s")
    print(f"成功请求：{s['req']}")
    print(f"总生成Token：{s['tokens']}")
    print(f"QPS：{s['qps']} req/s")
    print(f"Token 吞吐：{s['throughput']} token/s")
    print("-"*50)
    print(f"平均 TTFT：{s['avg_ttft']} ms")
    print(f"平均 TPOT：{s['avg_tpot']} ms/token")
    print(f"平均总延迟：{s['avg_lat']} ms")
    print("-"*50)
    print("📊 TTFT 首包延迟 百分位")
    print(f"P50：{s['ttft_p50']} ms")
    print(f"P90：{s['ttft_p90']} ms")
    print(f"P95：{s['ttft_p95']} ms")
    print(f"P99：{s['ttft_p99']} ms")
    print("-"*50)
    print("📊 总请求延迟 百分位")
    print(f"P50：{s['lat_p50']} ms")
    print(f"P90：{s['lat_p90']} ms")
    print(f"P95：{s['lat_p95']} ms")
    print(f"P99：{s['lat_p99']} ms")
    print("="*70)