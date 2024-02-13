#!/usr/bin/env python3

""" Утилита для работы с кодами статуса.

Сделано на основе uvicorn/logging.py:AccessFormatter (BSD 3-Clause License).
"""

from http import HTTPStatus
from typing import Any, Callable

from runtime.lib.decorators import singleton
from runtime.lib.logging.decorators import traced


@singleton
class StatusCodeUtility(object):
  """ Утилита для работы с кодами статуса.
  """

  @traced
  def __init__(self):
    """ Инициализирует класс.
    """

    self.__status_code_colours: dict[int, Any] = {
      1: lambda code: code,
      2: lambda code: f"\u001b[32m{code}\u001b[0m",
      3: lambda code: f"\u001b[33m{code}\u001b[0m",
      4: lambda code: f"\u001b[91m{code}\u001b[0m",
      5: lambda code: f"\u001b[31m{code}\u001b[0m"
    }

  @traced
  def get_status_code(self, status_code: int) -> str:
    """ Получает код и фразу.

    :param status_code: Код статуса.

    :return: Код с фразой.
    """

    try:
      status_phrase: str = HTTPStatus(status_code).phrase
    except ValueError:
      status_phrase: str = ""

    status_and_phrase: str = f"{status_code} {status_phrase}"

    def default(code: str) -> str:
      """ Получает код и фразу по умолчанию.

      :param code: Код статуса.

      :return: Код с фразой.
      """

      return status_and_phrase

    func: Callable = self.__status_code_colours.get(status_code // 100, default)

    return func(status_and_phrase)
