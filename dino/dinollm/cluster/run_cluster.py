"""
distributed/run_cluster.py
启动本地分布式推理集群：N个工作节点 + 1个路由节点。
每个工作节点独立加载模型（模拟多机器/多进程）。
路由器负责健康检查 + 轮询转发请求。

使用方法：
    python distributed/run_cluster.py --workers 2 --base-port 8001 --router-port 8080

然后请求路由器：
    curl -X POST http://localhost:8080/generate
         -H 'Content-Type: application/json'
         -d '{"prompt": "Hello, distributed world!", "max_new_tokens": 30}'

查看节点统计：
    curl http://localhost:8080/stats
"""
# 命令行参数解析
import argparse
# 子进程：用来启动worker和router服务
import subprocess
# 系统相关：获取当前Python解释器路径
import sys
# 时间：等待、超时
import time
# 环境变量
import os
# 信号：处理Ctrl+C关闭进程
import signal

# HTTP请求库，用于检查服务健康状态
import requests


def wait_for_health(url: str, max_wait: int = 60) -> bool:
    """
    循环访问 /health 接口，直到服务启动成功或超时
    参数：
        url: 服务地址（不带/health）
        max_wait: 最大等待秒数，默认60秒
    返回：
        True=健康 / False=超时失败
    """
    # 计算截止时间
    deadline = time.time() + max_wait
    # 在超时前不断重试
    while time.time() < deadline:
        try:
            # 访问健康检查接口，2秒超时
            resp = requests.get(f"{url}/health", timeout=2)
            # 返回200说明服务正常
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            # 连接失败（服务还没启动），忽略错误
            pass
        # 每秒重试一次
        time.sleep(1)
    # 超时都没启动成功
    return False


def launch_cluster(num_workers: int, base_port: int, router_port: int):
    """
    核心函数：启动整个分布式集群
    参数：
        num_workers: 工作节点数量
        base_port: worker起始端口号（依次递增）
        router_port: 路由端口
    """
    procs = []          # 保存所有子进程（worker + router）
    worker_urls = []    # 保存所有worker地址，传给router

    # 打印启动标题
    print(f"\n{'='*55}")
    print(f"  Launching ML Inference Cluster")
    print(f"  Workers: {num_workers}  |  Router: :{router_port}")
    print(f"{'='*55}\n")

    # ===================== 1. 批量启动所有Worker =====================
    for i in range(num_workers):
        # 端口依次递增：8001、8002、8003...
        port = base_port + i
        # 给worker起名字：worker-1、worker-2...
        worker_id = f"worker-{i+1}"
        # 拼接地址
        url = f"http://localhost:{port}"
        # 存入地址列表
        worker_urls.append(url)

        # 复制系统环境变量
        env = os.environ.copy()
        # 给worker设置环境变量，方便内部识别
        env["WORKER_ID"] = worker_id
        env["WORKER_PORT"] = str(port)

        print(f"[Cluster] Starting {worker_id} on port {port}...")
        # 用子进程启动：python -m distributed.worker
        proc = subprocess.Popen(
            [sys.executable, "-m","dinollm", "--port", str(port), "--model", "/root/autodl-tmp/model/Qwen3-0.6B","--cuda-graph-max-bs","0"],
            env=env
        )
        # 把进程加入列表，方便后面统一关闭
        procs.append(proc)

    # ===================== 2. 等待所有Worker启动成功 =====================
    print("\n[Cluster] Waiting for workers to become healthy...")
    for url in worker_urls:
        # 调用上面的健康检查函数
        if wait_for_health(url):
            print(f"  ✔ {url} — healthy")
        else:
            print(f"  ✗ {url} — FAILED to start within timeout")

    # ===================== 3. 启动路由服务 =====================
    print(f"\n[Cluster] Starting router on port {router_port}...")
    worker_args = []
    # 把所有worker地址拼成命令行参数：--workers http://localhost:8001 --workers http://localhost:8002 ...
    for url in worker_urls:
        worker_args += ["--workers", url]

    # 启动router进程
    router_proc = subprocess.Popen(
        [sys.executable, "-m", "dinollm.cluster.router",
         "--port", str(router_port)] + worker_args
    )
    procs.append(router_proc)

    # 检查router是否启动成功
    if wait_for_health(f"http://localhost:{router_port}"):
        print(f"  ✔ Router — healthy at http://localhost:{router_port}")
    else:
        print(f"  ✗ Router failed to start")

    # ===================== 4. 打印集群使用说明 =====================
    print(f"\n{'='*55}")
    print(f"  Cluster is UP!")
    print(f"{'='*55}")
    print(f"  Router (send requests here): http://localhost:{router_port}/generate")
    print(f"  Worker stats:                http://localhost:{router_port}/stats")
    for url in worker_urls:
        print(f"  Worker health:               {url}/health")
    print(f"\n  Press Ctrl+C to shut down the cluster.\n")

    # ===================== 5. 保持运行，处理Ctrl+C关闭 =====================
    try:
        # 等待所有进程（阻塞，直到按Ctrl+C）
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        # 用户按Ctrl+C
        print("\n[Cluster] Shutting down...")
        # 终止所有worker和router
        for proc in procs:
            proc.terminate()
        # 等待进程彻底退出
        for proc in procs:
            proc.wait(timeout=5)
        print("[Cluster] Stopped.")


if __name__ == "__main__":
    # 解析启动命令参数
    parser = argparse.ArgumentParser(description="Launch distributed ML inference cluster")
    # --workers：指定worker数量，默认2
    parser.add_argument("--workers", type=int, default=1, help="Number of worker nodes")
    # --base-port：worker起始端口，默认8001
    parser.add_argument("--base-port", type=int, default=1919, help="Starting port for workers")
    # --router-port：路由端口，默认8080
    parser.add_argument("--router-port", type=int, default=2000, help="Router port")
    args = parser.parse_args()

    # 启动集群
    launch_cluster(args.workers, args.base_port, args.router_port)