#!/usr/bin/env python3

""" Функции для работы с данными.
"""

from pickle import load, dump
from typing import Any

from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.lib.decorators import singleton


@singleton
class DataUtility(object):
  """ Класс для работы с данными.
  """

  @traced
  def __init__(self) -> None:
    """ Инициализирует класс.
    """

    self.__logger: Logger = Logger(name="Lib/DataUtils")

  @traced
  def load_data(self, filename: str) -> Any:
    """ Загружает данные из дампа.

    :param filename: Имя файла.

    :return: Данные.
    """

    self.__logger.debug(f"Loading the data dump from \"{filename}\"…")

    data: Any = load(file=open(file=filename, mode="rb"))

    self.__logger.debug("Data dump loaded.")

    return data

  @traced
  def save_data(self, filename: str, data: Any) -> None:
    """ Сохраняет данные в дамп.

    :param filename: Имя файла.
    :param data: Данные для сохранения.

    :return: Данные.
    """

    self.__logger.debug(f"Saving the data dump to \"{filename}\"…")

    dump(obj=data, file=open(file=filename, mode="wb"))

    self.__logger.debug("Data dump saved.")
