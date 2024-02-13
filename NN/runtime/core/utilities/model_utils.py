#!/usr/bin/env python3

""" Функции для работы с моделями Cortex.
"""

import os
from typing import Any

from keras import Model
from tensorflow.python.keras.saving.save import load_model

from runtime.lib.decorators import singleton
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger


@singleton
class ModelUtils(object):
  """ Класс для работы с моделями.
  """

  @traced
  def __init__(self) -> None:
    """ Инициализирует класс.
    """

    self.__logger: Logger = Logger(name="Core/ModelUtils")

  @traced
  def load_model(self, model_path: str) -> Any:
    """ Загружает модель сегмента ядра Cortex.

    :param model_path: Путь к модели.

    :return: Модель.
    """

    self.__logger.debug(f"Loading a Cortex model from \"{model_path}\"…")

    if not os.path.exists(f"{model_path}/trained") or not os.path.exists(f"{model_path}/metadata.pkl"):
      self.__logger.error(f"Your \"{model_path}\" model is not trained. Train the model.")

      raise FileNotFoundError(f"The model directory does not exist.")

    model: Any = load_model(f"{model_path}/trained")

    self.__logger.debug("The model is loaded.")

    return model

  @traced
  def save_model(self, model_path: str, model: Model) -> None:
    """ Сохраняет модель сегмента ядра Cortex.

    :param model_path: Путь к модели.
    :param model: Объект модели.
    """

    self.__logger.debug(f"Saving the Cortex model to \"{model_path}\"…")

    model.save(filepath=f"{model_path}/trained")

    self.__logger.debug("The model is saved.")
