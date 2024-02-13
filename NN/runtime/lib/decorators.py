#!/usr/bin/env python3

""" Вспомогательные декораторы.
"""

from functools import wraps


def singleton(orig_cls):
  """ Делает из класса синглтон.

  :param orig_cls: Начальный класс.

  :return: Экземпляр класса.
  """

  orig_new = orig_cls.__new__
  instance = None

  @wraps(orig_cls.__new__)
  def __new__(cls, *args, **kwargs):
    """ Оборачивает класс.

    :param cls: Класс.
    :param args: Аргументы.
    :param kwargs: Именованные аргументы.

    :return: Экземпляр класса.
    """

    nonlocal instance

    if instance is None:
      instance = orig_new(cls, *args, **kwargs)

    return instance

  orig_cls.__new__ = __new__

  return orig_cls
