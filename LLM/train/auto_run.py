import os
import time
import subprocess
import datetime
import sys

# =========================
# 配置区域
# =========================

CHECK_INTERVAL = 30  # 每30秒检查一次
MIN_FREE_MEMORY = 24000  # 至少要求 18GB (18000MB) 空闲显存

# 将命令改写为列表形式，这是 subprocess 更推荐的安全写法
TRAIN_CMD = [
    sys.executable,
    "align_sid.py"
]


# =========================
# 辅助函数
# =========================

def log(message):
    """带时间戳的标准日志输出"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")


def get_gpu_info():
    """获取 GPU 显存信息，带有异常处理机制"""
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits"
            ],
            stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        log(f"获取 GPU 信息失败 (驱动可能繁忙): {e.output.decode().strip()}")
        return []

    gpus = []
    for line in result.strip().split("\n"):
        if not line:
            continue
        idx, total, used = map(int, line.split(","))
        free = total - used

        gpus.append({
            "index": idx,
            "total": total,
            "used": used,
            "free": free
        })

    return gpus


def find_available_gpu():
    """寻找满足条件的显存最大的可用 GPU"""
    gpus = get_gpu_info()
    if not gpus:
        return None, None

    # 找空闲显存最大的 GPU
    best_gpu = max(gpus, key=lambda x: x["free"])

    if best_gpu["free"] >= MIN_FREE_MEMORY:
        return best_gpu["index"], best_gpu["free"]

    return None, None


# =========================
# 主监控循环
# =========================

def main():
    log("启动排队监控系统...")
    log(f"目标配置: 至少需要 {MIN_FREE_MEMORY} MB 空闲显存 | 每 {CHECK_INTERVAL} 秒刷新一次")

    waiting = True

    try:
        while True:
            gpu_id, free_mem = find_available_gpu()

            if gpu_id is not None:
                # 换行清空之前的等待状态
                if waiting: print()

                log(f"成功锁定可用 GPU: {gpu_id} (当前空闲显存: {free_mem} MB)")
                log(f"开始启动训练任务: {' '.join(TRAIN_CMD)}")

                # 复制当前环境变量，并仅为子进程单独注入 CUDA_VISIBLE_DEVICES
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                # 启动训练进程 (标准输出会实时打印在屏幕上)
                start_time = time.time()
                process = subprocess.run(TRAIN_CMD, env=env)
                end_time = time.time()

                duration = datetime.timedelta(seconds=int(end_time - start_time))

                # 检查任务是否成功完成
                if process.returncode == 0:
                    log(f"训练任务圆满结束！(耗时: {duration})")
                else:
                    log(f"训练任务异常退出，返回码: {process.returncode} (耗时: {duration})")

                # 运行完毕后退出监控脚本
                break

            else:
                # 使用 \r 回车符在同一行刷新状态，避免终端被日志刷屏
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                sys.stdout.write(f"\r[{now}]暂无满足条件 ({MIN_FREE_MEMORY} MB) 的 GPU，继续等待...")
                sys.stdout.flush()
                waiting = True

                time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print()  # 捕获 Ctrl+C 换行
        log("收到用户中断信号 (Ctrl+C)，已停止排队监控。")


if __name__ == "__main__":
    main()