#!/usr/bin/env python3

""" Ответ API CDR.
"""

from runtime.lib.abstract.dict_class import DictClass
from runtime.lib.logging.decorators import traced


# После изменения необходимо проверить соответствие со схемой ../schemes/response_scheme.py:DialogueReplyingResponseScheme.


class DialogueReplyingResponse(DictClass):
  """ Ответ API CDR.
  """

  @traced
  def __init__(self, response: str) -> None:
    """ Инициализирует класс.

    :param response: Текст ответа.
    """

    self.response: str = response
