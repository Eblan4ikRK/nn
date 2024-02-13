#!/usr/bin/env python3

""" Каналы логирования.
"""


class LogChannels(object):
  """ Каналы логирования.
  """

  # Критическая ошибка.
  critical: dict[str, str | int] = {
    "name": "CRITICAL",
    "color_code": "\u001b[31m", # "\u001b[41m",
    "level": 1
  }

  # Ошибка.
  error: dict[str, str | int] = {
    "name": "ERROR",
    "color_code": "\u001b[91m",
    "level": 2
  }

  # Предупреждение.
  warning: dict[str, str | int] = {
    "name": "WARNING",
    "color_code": "\u001b[33m",
    "level": 3
  }

  # Информационные сообщения.
  info: dict[str, str | int] = {
    "name": "INFO",
    "color_code": "",
    "level": 4
  }

  # Информационные сообщения (уведомления).
  notice: dict[str, str | int] = {
    "name": "NOTICE",
    "color_code": "\u001b[32m",
    "level": 5
  }

  # Отладочная информация.
  debug: dict[str, str | int] = {
    "name": "DEBUG",
    "color_code": "\u001b[37;2m",
    "level": 6
  }

  # Вызовы функций.
  trace: dict[str, str | int] = {
    "name": "TRACE",
    "color_code": "\u001b[30m",
    "level": 7
  }
