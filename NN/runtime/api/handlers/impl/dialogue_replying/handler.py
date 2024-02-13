#!/usr/bin/env python3

""" Обработчик роутеров для взаимодействия с CDR.
"""

from typing import Any

from fastapi import APIRouter

from runtime.api.handlers.handler_base import Handler
from runtime.api.handlers.impl.dialogue_replying.objects.response import DialogueReplyingResponse
from runtime.api.handlers.impl.dialogue_replying.schemes.request_scheme import DialogueReplyingRequestScheme
from runtime.api.handlers.impl.dialogue_replying.schemes.response_scheme import DialogueReplyingResponseScheme
from runtime.core.segments.impl.dialogue_replying.inferencer import DialogueReplyingInferencer
from runtime.lib.decorators import singleton
from runtime.lib.logging.decorators import traced


@singleton
class DialogueReplyingHandler(Handler):
  """ Класс обработчика роутеров CDR.
  """

  segment_name: str = "dialogue_replying"

  @traced
  def create_router(self, model_path: str, path: str, summary: str) -> APIRouter:
    """ Создаёт роутер диалоговой системы (CDR).

    :param model_path: Путь к директории модели диалоговой системы.
    :param path: Путь роутера.
    :param summary: Описание роутера.

    :return: Экземпляр роутера CDR.
    """

    inferencer: DialogueReplyingInferencer = DialogueReplyingInferencer(model_path=model_path)

    router: APIRouter = APIRouter()

    @router.post(
      path=f"/{self.segment_name}{path}",
      response_model=DialogueReplyingResponseScheme,
      summary=summary,
      tags=[self.segment_name]
    )
    def handle(request: DialogueReplyingRequestScheme) -> dict[str, Any]:
      """ Отвечает на запросы.

      :param request: Запрос.

      :return: Объект ответа.
      """

      result: str = inferencer.inference(query=request.query)

      response: DialogueReplyingResponse = DialogueReplyingResponse(response=result)

      return response.to_dict()

    return router
