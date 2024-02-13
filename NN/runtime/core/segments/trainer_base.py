#!/usr/bin/env python3

""" Шаблон компонента для обучения нейросети.
"""

from abc import abstractmethod


class Trainer(object):
  """ Шаблон класса для обучения нейросети.
  """

  @abstractmethod
  def __init__(self, *args) -> None:
    """ Шаблон функции инициализации класса.
    """

    pass

  @abstractmethod
  def train(self, *args) -> None:
    """ Шаблон функции запуска обучения нейросети.
    """

    pass
