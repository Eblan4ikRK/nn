#!/usr/bin/env python3

""" Ответ API CCS.
"""

from runtime.lib.abstract.dict_class import DictClass
from runtime.lib.logging.decorators import traced


# После изменения необходимо проверить соответствие со схемой ../schemes/response_scheme.py:CaptchaSolvingResponseScheme.


class CaptchaSolvingResponse(DictClass):
  """ Ответ API CCS.
  """

  @traced
  def __init__(self, response: str) -> None:
    """ Инициализирует класс.

    :param response: Обнаруженные на изображении символы.
    """

    self.response: str = response
