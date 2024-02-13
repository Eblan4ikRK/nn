#!/usr/bin/env python3

""" Менеджер обработчиков роутеров сегментов.
"""

from runtime.api.handlers.impl.captcha_solving.handler import CaptchaSolvingHandler
from runtime.api.handlers.impl.dialogue_replying.handler import DialogueReplyingHandler
from runtime.api.handlers.handler_base import Handler
from runtime.lib.decorators import singleton
from runtime.lib.logging.decorators import traced


@singleton
class HandlerManager(object):
  """ Менеджер обработчиков роутеров сегментов.
  """

  @traced
  def __init__(self) -> None:
    """ Инициализирует класс.
    """

    self.handlers: list[Handler] = [
      CaptchaSolvingHandler(),
      DialogueReplyingHandler()
    ]

  @traced
  def get_handler(self, segment_name: str) -> Handler:
    """ Получает обработчик роутеров сегмента по названию сегмента.

    :param segment_name: Имя сегмента.

    :return: Обработчик роутеров сегмента.

    :raise NotImplementedError: Ошибка, получаемая при отсутствии сегмента с указанным названием в списке обработчиков роутеров.
    """

    for handler in self.handlers:
      if handler.segment_name == segment_name:
        return handler

    raise NotImplementedError(f"The \"{segment_name}\" segment router handler is not implemented in Cortex Runtime.")
