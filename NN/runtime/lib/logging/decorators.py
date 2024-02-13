#!/usr/bin/env python3

""" Декораторы для логирования.
"""

import time
from typing import Callable, Any

from runtime.lib.logging.helpers.static_logger import StaticLogger
from runtime.lib.utilities.env_utility import EnvUtility


def traced(func: Any) -> Callable:
  """ Логирует вызов функции (в TRACE лог).

  :param func: Функция.

  :return: Функция.
  """

  def wrapper(*args, **kwargs) -> Callable:
    """ Оборачивает и логирует вызов функции.

    :param args: Аргументы функции.
    :param kwargs: Именованные аргументы функции.

    :return: Функция.
    """

    start_time: float = time.time()

    result = func(*args, **kwargs)

    exec_time: float = (time.time() - start_time) * 10 ** 3

    args_tuple: tuple[str] = func.__code__.co_varnames
    joined_args: str = ", ".join(list(args_tuple))

    execution_time_string = f" [{exec_time:.03f}ms]" if EnvUtility().get_bool(var_name="CORTEX_LIB_LOG_TRACE_EXEC_TIME") else ""
    arguments_string = f"({joined_args})" if EnvUtility().get_bool(var_name="CORTEX_LIB_LOG_TRACE_ARGNAMES") else ""

    StaticLogger().trace(f"{func.__module__}:{func.__qualname__}{arguments_string}{execution_time_string}.")

    return result

  return wrapper
