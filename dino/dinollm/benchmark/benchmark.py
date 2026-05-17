import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict


# ====================== 配置区 ======================
API_URL = "http://localhost:2000/v1/chat/completions"
API_KEY = "dummy"
PROMPT = "你好，请简单介绍一下自己"
CONCURRENCY = 10  # 并发数
TOTAL_REQUESTS = 100  # 总请求数
# ====================================================


class BenchmarkMetrics:
    """压测指标收集"""
    def __init__(self):
        self.ttft_list: List[float] = []
        self.total_latency_list: List[float] = []
        self.tpot_list: List[float] = []
        self.total_tokens = 0

    def add(self, ttft: float, total_latency: float, token_count: int):
        self.ttft_list.append(ttft)
        self.total_latency_list.append(total_latency)
        if token_count > 1:
            tpot = (total_latency - ttft) / (token_count - 1)
            self.tpot_list.append(tpot)
        self.total_tokens += token_count

    def compute(self) -> Dict:
        def pct(data, p):
            if not data: return 0.0
            return sorted(data)[int(len(data) * p / 100)]
        return {
            "ttft_ms": {
                "avg": round(statistics.mean(self.ttft_list), 2) if self.ttft_list else 0,
                "p50": round(pct(self.ttft_list, 50), 2),
                "p99": round(pct(self.ttft_list, 99), 2),
            },
            "total_latency_ms": {
                "avg": round(statistics.mean(self.total_latency_list), 2) if self.total_latency_list else 0,
                "p50": round(pct(self.total_latency_list, 50), 2),
                "p99": round(pct(self.total_latency_list, 99), 2),
            },
            "tpot_ms": {
                "avg": round(statistics.mean(self.tpot_list), 4) if self.tpot_list else 0,
                "p50": round(pct(self.tpot_list, 50), 4),
                "p99": round(pct(self.tpot_list, 99), 4),
            },
            "total_tokens": self.total_tokens,
            "total_requests": len(self.ttft_list),
        }


async def single_request(session: aiohttp.ClientSession, user_id: str):
    """单个流式请求，精确统计 TTFT / 总耗时 / token数"""
    start = time.perf_counter()
    ttft = None
    token_count = 0

    try:
        async with session.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "default",
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": True,
                "user": user_id  # 关键：测试粘性会话
            }
        ) as resp:
            async for line in resp.content:
                if not ttft:
                    ttft = (time.perf_counter() - start) * 1000
                token_count += 1

        total_latency = (time.perf_counter() - start) * 1000
        return ttft, total_latency, token_count

    except Exception as e:
        print(f"请求失败: {e}")
        return None, None, 0


async def worker(queue: asyncio.Queue, metrics: BenchmarkMetrics):
    async with aiohttp.ClientSession() as session:
        while True:
            user_id = await queue.get()
            try:
                ttft, total_lat, tokens = await single_request(session, user_id)
                if ttft:
                    metrics.add(ttft, total_lat, tokens)
            finally:
                queue.task_done()


async def main():
    print("=" * 60)
    print("🚀 LLM 路由压测脚本 (支持流式 + 粘性会话 + TTFT/TPOT/P99)")
    print(f"🌐 接口: {API_URL}")
    print(f"👥 并发: {CONCURRENCY}")
    print(f"📦 总请求: {TOTAL_REQUESTS}")
    print(f"💬 提示词: {PROMPT}")
    print("=" * 60)

    queue = asyncio.Queue(maxsize=TOTAL_REQUESTS)
    metrics = BenchmarkMetrics()

    # 启动并发 worker
    for _ in range(CONCURRENCY):
        asyncio.create_task(worker(queue, metrics))

    # 填充任务（不同 user_id 测试粘性）
    for i in range(TOTAL_REQUESTS):
        user_id = f"test_user_{i % 10}"  # 模拟 10 个用户循环
        queue.put_nowait(user_id)

    # 等待全部完成
    await queue.join()

    # 输出结果
    print("\n✅ 压测完成！\n")
    res = metrics.compute()
    print(f"📊 总请求数: {res['total_requests']}")
    print(f"📝 总生成token: {res['total_tokens']}")
    print("\n🔥 TTFT (首token时延 ms)")
    print(f"  Avg: {res['ttft_ms']['avg']} | P50: {res['ttft_ms']['p50']} | P99: {res['ttft_ms']['p99']}")
    print("\n⏱️ 总时延 ms")
    print(f"  Avg: {res['total_latency_ms']['avg']} | P50: {res['total_latency_ms']['p50']} | P99: {res['total_latency_ms']['p99']}")
    print("\n⚡ TPOT (per token 时延 ms)")
    print(f"  Avg: {res['tpot_ms']['avg']} | P50: {res['tpot_ms']['p50']} | P99: {res['tpot_ms']['p99']}")
    print("\n✅ 压测结束！")


if __name__ == "__main__":
    asyncio.run(main())