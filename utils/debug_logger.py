"""
调试日志相关工具
"""

import logging
from functools import wraps

class DebugLogger:
    """调试日志记录器"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def log(self, level, message):
        """记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
        """
        if self.debug:
            if level == 'error':
                self.logger.error(message)
            elif level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            else:
                self.logger.debug(message)

    def log_entry(self, func_name, args=None, kwargs=None):
        """记录函数入口日志
        
        Args:
            func_name: 函数名
            args: 位置参数
            kwargs: 关键字参数
        """
        if self.debug:
            args_str = f", args: {args}" if args else ""
            kwargs_str = f", kwargs: {kwargs}" if kwargs else ""
            self.logger.info(f"Entering {func_name}{args_str}{kwargs_str}")

    def log_exit(self, func_name, result=None):
        """记录函数退出日志
        
        Args:
            func_name: 函数名
            result: 返回结果
        """
        if self.debug:
            result_str = f", returned: {result}" if result is not None else ""
            self.logger.info(f"Exiting {func_name}{result_str}")

def log_method(debug=True):
    """方法日志装饰器
    
    Args:
        debug: 是否开启调试日志
        
    Returns:
        装饰器函数
    """
    logger = DebugLogger(debug=debug)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log_entry(func.__name__, args, kwargs)
            try:
                result = func(*args, **kwargs)
                logger.log_exit(func.__name__, result)
                return result
            except Exception as e:
                logger.log('error', f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator 