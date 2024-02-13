#!/usr/bin/env python3

""" Функции для работы со строками.
"""

from difflib import SequenceMatcher

from runtime.lib.decorators import singleton
from runtime.lib.logging.decorators import traced


@singleton
class StringUtility(object):
  """ Функции для работы со строками.
  """

  @staticmethod
  @traced
  def get_strings_similarity(a: str, b: str) -> float:
    """ Получает процент схожести строк (дистанцию Левенштейна).

    :param a: Первая строка.
    :param b: Вторая строка.

    :return: Значение схожести (%).
    """

    sequence_matcher: SequenceMatcher = SequenceMatcher(a=a, b=b)

    multiplied_result: float = sequence_matcher.ratio() * 100

    return multiplied_result
