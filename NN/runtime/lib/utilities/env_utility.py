#!/usr/bin/env python3

""" Функции для работы с переменными окружения.
"""

import os
from functools import cache
from typing import Any

from runtime.lib.decorators import singleton


@singleton
class EnvUtility(object):
  """ Класс для работы с переменными окружения.
  """

  @staticmethod
  def __resolve_value(var_name: str, default_value: Any) -> Any:
    """ Получает значение переменной окружения.

    В случае отсутствия, возвращает значение по умолчанию.

    :param var_name: Имя переменной окружения.
    :param default_value: Значение по умолчанию.

    :return: Значение переменной, либо значение по умолчанию.
    """

    value: Any = default_value if var_name not in os.environ else os.environ[var_name]

    return value

  @cache
  def get_bool(self, var_name: str, default_value: bool = False) -> bool:
    """ Получает переменную окружения (тип: Логический).

    Если переменная отсутствует, то возвращает значение по умолчанию.

    :param var_name: Имя переменной окружения.
    :param default_value: Значение по умолчанию.

    :return: Значение переменной, либо значение по умолчанию.
    """

    value: str = str(self.__resolve_value(var_name=var_name, default_value=default_value))

    if value in ["true", "True"]:
      return True

    if value in ["false", "False"]:
      return False

    raise ValueError(f"The value of the variable \"{var_name}\" must be an boolean.")

  @cache
  def get_int(self, var_name: str, default_value: int = 0) -> int:
    """ Получает переменную окружения (тип: Целое число).

    Если переменная отсутствует, то возвращает значение по умолчанию.

    :param var_name: Имя переменной окружения.
    :param default_value: Значение по умолчанию.

    :return: Значение переменной, либо значение по умолчанию.
    """

    value: str = self.__resolve_value(var_name=var_name, default_value=default_value)

    try:
      new_value: int = int(value)

      return new_value
    except ValueError:
      raise ValueError(f"The value of the variable \"{var_name}\" must be an integer.")

  @cache
  def get_str(self, var_name: str, default_value: str = "") -> str:
    """ Получает переменную окружения (тип: Строка).

    Если переменная отсутствует, то возвращает значение по умолчанию.

    :param var_name: Имя переменной окружения.
    :param default_value: Значение по умолчанию.

    :return: Значение переменной, либо значение по умолчанию.
    """

    value: str = self.__resolve_value(var_name=var_name, default_value=default_value)

    return value

  @cache
  def get_list(self, var_name: str, default_value: list[Any]) -> list[Any]:
    """ Получает переменную окружения (тип: Список).

    Если переменная отсутствует, то возвращает значение по умолчанию.

    :param var_name: Имя переменной окружения.
    :param default_value: Значение по умолчанию.

    :return: Значение переменной, либо значение по умолчанию.
    """

    value: list[Any] = self.__resolve_value(var_name=var_name, default_value=tuple(default_value)).split()

    return value
