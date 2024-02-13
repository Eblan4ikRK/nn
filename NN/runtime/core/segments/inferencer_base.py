#!/usr/bin/env python3

""" Шаблон компонента для вывода данных.
"""

from abc import abstractmethod
from typing import Any


class Inferencer(object):
  """ Шаблон класса для вывода данных.
  """

  @abstractmethod
  def __init__(self, *args) -> None:
    """ Шаблон функции инициализации класса для вывода данных.
    """

    pass

  @abstractmethod
  def inference(self, *args) -> Any:
    """ Шаблон функции вывода данных.

    :return: Выходные данные.
    """

    pass
