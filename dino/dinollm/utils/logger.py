# 固定写法：让类型提示更稳定，不用管原理
from __future__ import annotations

# 偏函数：用来绑定参数，简化调用
from functools import partial
# 类型检查专用标记，新手忽略
from typing import TYPE_CHECKING

# 全局日志级别变量，一开始是空的
_LOG_LEVEL = None


def init_logger(
    name: str,          # 日志名字
    suffix: str = "",   # 后缀标识
    *,
    strip_file: bool = True,  # 是否简化文件名
    level: str | None = None, # 日志级别（DEBUG/INFO等）
    use_pid: bool | None = None,    # 是否显示进程ID
    use_tp_rank: bool | None = None,# 是否显示分布式rank
):
    """
    初始化日志系统
    功能：带颜色、漂亮格式、分布式多卡只打印主卡日志
    """
    # 延迟导入日志、系统、操作系统库
    import logging
    import os
    import sys

    # 声明使用全局的日志级别变量
    global _LOG_LEVEL

    # 如果日志级别还没初始化
    if _LOG_LEVEL is None:
        # 日志级别映射表
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,      # 调试（最详细）
            "INFO": logging.INFO,        # 普通信息
            "WARNING": logging.WARNING,  # 警告
            "ERROR": logging.ERROR,      # 错误
            "CRITICAL": logging.CRITICAL,# 致命错误
        }

        # 优先级：传入level > 环境变量LOG_LEVEL > 默认INFO
        level = level or os.getenv("LOG_LEVEL", "").upper()
        # 设置最终日志级别，默认INFO
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    # 如果需要简化路径，只保留文件名
    if strip_file:
        suffix = os.path.basename(suffix)

    # 如果有后缀，加个分隔符 |
    if suffix:
        suffix = f"|{suffix}"

    # 是否显示进程ID（从环境变量LOG_PID读取）
    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    # 如果需要显示PID，拼接到后缀里
    if use_pid:
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    # 分布式张量并行信息（多卡训练/推理用）
    tp_info = None


    # ====================== 核心：彩色日志格式类 ======================
    class ColorFormatter(logging.Formatter):
        """带颜色的日志格式化器，让日志好看又清晰"""

        # ANSI 颜色代码（控制台颜色）
        COLORS = {
            "DEBUG": "\033[36m",    # 青色
            "INFO": "\033[32m",     # 绿色
            "WARNING": "\033[33m",  # 黄色
            "ERROR": "\033[31m",    # 红色
            "CRITICAL": "\033[35m", # 紫色
        }
        RESET = "\033[0m"   # 恢复默认颜色
        BOLD = "\033[1m"    # 加粗

        def format(self, record):
            # 动态导入：获取分布式多卡信息
            from dinollm.distributed import try_get_tp_info

            # 格式化时间戳：[年-月-日|时:分:秒|后缀]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            
            # 缓存多卡信息，避免重复获取
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            
            # 如果是多卡，并且允许显示卡ID，加上卡序号
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            
            # 把后缀塞进时间格式里
            timestamp = timestamp.format(suffix=real_suffix)

            # 根据日志级别拿对应颜色
            level_color = self.COLORS.get(record.levelname, "")

            # 给日志级别加上颜色并固定宽度
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            # 日志正文内容
            message = record.getMessage()

            # 最终输出格式：[时间+后缀] 颜色级别 消息
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"


    # ====================== 创建并配置logger ======================
    # 获取指定名称的日志器
    logger = logging.getLogger(name)
    # 设置日志级别
    logger.setLevel(_LOG_LEVEL)

    # 清空已有的处理器，避免重复打印
    logger.handlers.clear()

    # 创建输出到控制台的处理器
    handler = logging.StreamHandler(sys.stdout)
    # 使用我们自定义的彩色格式化器
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    # 把处理器加到日志器
    logger.addHandler(handler)

    # 禁止向上传递日志，避免重复输出
    logger.propagate = False


    # ====================== 分布式多卡专用：只让主卡打印日志 ======================
    def _call_rank0(msg, *args, _which, **kwargs):
        # 获取多卡信息
        from dinollm.distributed import get_tp_info
        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        
        # 必须先设置好多卡信息
        assert tp_info is not None, "TP info not set yet"
        
        # 如果是主卡（rank=0），才打印日志
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)


    # ====================== 类型提示包装（编辑器友好，新手忽略） ======================
    if TYPE_CHECKING:
        class WrapperLogger(logging.Logger):
            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...
        return WrapperLogger(name)

    # ====================== 给日志器添加 rank0 专用方法 ======================
    else:
        # 绑定：logger.info_rank0 → 只主卡打印info
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        
        # 返回配置好的日志器
        return logger