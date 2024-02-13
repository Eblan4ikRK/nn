#!/usr/bin/env python3

""" Обработчик роутеров для взаимодействия с CCS.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from runtime.api.handlers.handler_base import Handler
from runtime.api.handlers.impl.captcha_solving.objects.response import CaptchaSolvingResponse
from runtime.api.handlers.impl.captcha_solving.schemes.request_scheme import CaptchaSolvingRequestScheme
from runtime.api.handlers.impl.captcha_solving.schemes.response_scheme import CaptchaSolvingResponseScheme
from runtime.core.segments.impl.captcha_solving.inferencer import CaptchaSolvingInferencer
from runtime.lib.decorators import singleton
from runtime.lib.logging.decorators import traced


@singleton
class CaptchaSolvingHandler(Handler):
  """ Класс обработчика роутеров CCS.
  """

  segment_name: str = "captcha_solving"

  @traced
  def create_router(self, model_path: str, path: str, summary: str) -> APIRouter:
    """ Создаёт роутер для решения капчи (CCS).

    :param model_path: Путь к директории модели.
    :param path: Путь роутера.
    :param summary: Описание роутера.

    :return: Экземпляр роутера CCS.
    """

    inferencer: CaptchaSolvingInferencer = CaptchaSolvingInferencer(model_path=model_path)

    router: APIRouter = APIRouter()

    @router.post(
      path=f"/{self.segment_name}{path}",
      response_model=CaptchaSolvingResponseScheme,
      summary=summary,
      tags=[self.segment_name]
    )
    def handle(request: CaptchaSolvingRequestScheme) -> dict[str, Any]:
      """ Отвечает на запросы.

      :param request: Запрос.

      :return: Объект ответа.
      """

      try:
        result: str = inferencer.inference(query=request.query)

        response: CaptchaSolvingResponse = CaptchaSolvingResponse(response=result)

        return response.to_dict()
      except ValueError:
        raise HTTPException(status_code=422, detail="Invalid Base64 image.")

    return router
