#!/usr/bin/env python3

""" Базовый объект обработчика роутеров сегмента.
"""

from abc import abstractmethod

from fastapi import APIRouter


class Handler(object):
  """ Обработчик роутеров сегмента.
  """

  # Имя сегмента ядра, за роутеры которого отвечает обработчик.
  segment_name: str

  @abstractmethod
  def create_router(self, model_path: str, path: str, summary: str) -> APIRouter:
    """ Создает роутер FastAPI для использования модели сегмента.

    :param model_path: Путь к модели нейросети, которую представляет роутер.
    :param path: Путь роутера.
    :param summary: Описание роутера.

    :return: Экземпляр роутера FastAPI.
    """

    pass
