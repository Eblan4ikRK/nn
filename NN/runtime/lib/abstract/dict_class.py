#!/usr/bin/env python3

""" Класс с полями конвертируемыми в словарь.
"""

from typing import Any

from runtime.lib.logging.decorators import traced


class DictClass(object):
  """ Класс с полями конвертируемыми в словарь.
  """

  @traced
  def to_dict(self) -> dict[str, Any]:
    """ Конвертирует поля класса в словарь.

    Помимо стандартных поддерживает следующие типы:
    - DictClass.
    - list[DictClass].

    :return: Словарь.
    """

    fields: dict[str, Any] = self.__dict__

    fields_dict: dict[str, Any] = {}

    for field in fields:
      if type(fields[field]) is list and all(isinstance(item, DictClass) for item in fields[field]):
        sub_dict_classes: list[dict[str, Any]] = []

        for item in fields[field]:
          sub_dict_classes.append(item.to_dict())

        fields_dict[field] = sub_dict_classes
      elif isinstance(fields[field], DictClass):
        fields_dict[field] = fields[field].to_dict()
      else:
        fields_dict[field] = fields[field]

    # print(fields_dict)

    return fields_dict
