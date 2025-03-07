# async_utils.py
import asyncio
import threading
import functools
import logging
from contextlib import contextmanager

# 线程本地存储
_thread_local = threading.local()

@contextmanager
def managed_event_loop():
    """为当前线程创建和管理事件循环的上下文管理器"""
    # 检查当前线程是否已有事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        owned = True
    else:
        owned = False
    
    # 如果循环已关闭，创建新的
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        owned = True
    
    _thread_local.loop = loop
    try:
        yield loop
    finally:
        if owned:
            try:
                # 关闭待处理的任务
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logging.warning(f"Error closing pending tasks: {e}")

def run_async(coroutine_func):
    """装饰器：包装异步函数以便在同步环境中调用"""
    @functools.wraps(coroutine_func)
    def wrapper(*args, **kwargs):
        with managed_event_loop() as loop:
            return loop.run_until_complete(coroutine_func(*args, **kwargs))
    return wrapper

def ensure_future(coroutine, loop=None):
    """安全地将协程安排到事件循环中执行"""
    if loop is None:
        with managed_event_loop() as loop:
            return asyncio.ensure_future(coroutine, loop=loop)
    return asyncio.ensure_future(coroutine, loop=loop)