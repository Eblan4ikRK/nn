#!/usr/bin/env python3

""" Логер.
"""

from datetime import datetime
from typing import Any

from runtime.lib.logging.channels import LogChannels
from runtime.lib.utilities.env_utility import EnvUtility


class Logger(object):
  """ Логер.
  """

  def __init__(self, name: str = "Lib/Logger") -> None:
    """ Инициализирует логер.

    :param name: Имя логера.
    """

    self.__logger_name: str = name

  @staticmethod
  def __get_formatted_date() -> str:
    """ Получает строку с форматированной датой.

    :return: Строка с форматированной датой.
    """

    current_time: datetime = datetime.now()

    date: str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    microseconds: str = current_time.strftime("%f")[:3]

    return f"{date},{microseconds}"

  def __log(self, channel: dict[str, str | int], text: str) -> None:
    """ Логирует текст.

    :param channel: Канал логирования.
    :param text: Текст.
    """

    if EnvUtility().get_int(var_name="CORTEX_LIB_LOG_LEVEL", default_value=5) < int(channel["level"]):
      return None

    # if not self.__logger_name in get_list(var_name="CORTEX_CORE_LOGGERS", default_value=["Server"]):
    #   return None

    print(f"{channel['color_code']}{self.__get_formatted_date()} {channel['name']} [{self.__logger_name}]: {text}\u001b[0m")

  def info(self, text: Any = "") -> None:
    """ Логирует информацию.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.info, text=str(text))

  def warning(self, text: Any = "") -> None:
    """ Логирует предупреждения.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.warning, text=str(text))

  def error(self, text: Any = "") -> None:
    """ Логирует ошибки.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.error, text=str(text))

  def critical(self, text: Any = "") -> None:
    """ Логирует критические ошибки.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.critical, text=str(text))

  def notice(self, text: Any = "") -> None:
    """ Логирует уведомления.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.notice, text=str(text))

  def debug(self, text: Any = "") -> None:
    """ Логирует отладочную информацию.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.debug, text=str(text))

  def trace(self, text: Any = "") -> None:
    """ Логирует вызовы функций.

    :param text: Текст.
    """

    self.__log(channel=LogChannels.trace, text=str(text))
