#!/usr/bin/env python3

""" Компонент для вывода данных (CCS).

Сделано на основе https://keras.io/examples/vision/captcha_ocr.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import tensorflow as tf
from keras import Model
from keras.api import keras
from keras.layers import StringLookup
from numpy import ndarray
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.engine.functional import Functional

from runtime.core.segments.inferencer_base import Inferencer
from runtime.core.utilities.model_utils import ModelUtils
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.lib.utilities.data_utility import DataUtility
from runtime.lib.utilities.env_utility import EnvUtility


class CaptchaSolvingInferencer(Inferencer):
  """ Класс для вывода данных.
  """

  @traced
  def __init__(self, model_path: str) -> None:
    """ Функция инициализации класса для вывода данных.

    :param model_path: Путь к директории модели.
    """

    super().__init__()

    self.__logger: Logger = Logger("Core/CaptchaSolving/Inferencer")

    self.__logger.debug("Initializing the CCS inferencer instance…")

    # region ЗАГРУЗКА МОДЕЛИ.

    raw_model: Model = ModelUtils().load_model(model_path=model_path)

    # TODO: Протестировать сохранение подобной модели в готовом виде.
    self.__model: Functional = Functional(
      raw_model.get_layer(name="image").input,  # type: ignore
      raw_model.get_layer(name="dense2").output  # type: ignore
    )

    # endregion

    # region ЗАГРУЗКА МЕТАДАННЫХ.

    metadata: Any = DataUtility().load_data(filename=f"{model_path}/metadata.pkl")

    self.__characters: list[str] = metadata["characters"]
    self.__max_label_length: int = metadata["max_label_length"]
    self.__image_width: int = metadata["image_width"]
    self.__image_height: int = metadata["image_height"]

    # endregion

    self.__char_to_num: StringLookup = StringLookup(vocabulary=list(self.__characters))
    self.__num_to_char: StringLookup = StringLookup(vocabulary=self.__char_to_num.get_vocabulary(), invert=True)

  @traced
  def __decode_batch_predictions(self, predictions: ndarray) -> str:
    """ Декодирует выходные данные нейросети.

    :param predictions: Выходные данные (предсказания) модели.

    :return: Декодированные выходные данные.
    """

    self.__logger.debug("Prediction…")

    input_length: ndarray = np.ones(shape=predictions.shape[0]) * predictions.shape[1]

    results: list[Any] = keras.backend.ctc_decode(y_pred=predictions, input_length=input_length)[0][0][:, :self.__max_label_length]

    output_text: list[str] = []

    for result in results:
      result: str = tf.strings.reduce_join(inputs=self.__num_to_char(result)).numpy().decode(encoding="utf-8")

      output_text.append(result)

    result: str = output_text[0]

    # Пробелами при подготовке данных были дополнены метки файлов с длиной меньше максимальной.
    # Нераспознанные символы нейросеть отмечает как "[UNK]".
    if EnvUtility().get_bool(var_name="CORTEX_CORE_CCS_CLEAN_RESPONSE", default_value=True):
      result: str = result.replace("[UNK]", "").strip()

    self.__logger.debug(f"Prediction completed. Output: \"{result}\".")

    return result

  @traced
  def __transform_image(self, source: str) -> Tensor:
    """ Декодирует и трансформирует изображение до подходящей формы.

    :param source: Закодированное в Base64 изображение.

    :return: Трансформированное в тензор изображение.

    :raise ValueError: Ошибка при некорректном изображении.
    """

    self.__logger.debug("Image transformation…")

    # Чтение изображения.
    image: Tensor = tf.io.decode_base64(input=source)

    # Декодировка и конвертация в серые цвета.
    try:
      image: Tensor = tf.io.decode_png(contents=image, channels=1)
    except Exception:
      self.__logger.debug("Invalid Base64 image.")

      raise ValueError("Invalid Base64 image.")

    # Преобразование в float32 в диапазоне [0, 1].
    image: Tensor = tf.image.convert_image_dtype(image=image, dtype=tf.float32)

    # Изменение до нужного размера.
    image: Tensor = tf.image.resize(images=image, size=[self.__image_height, self.__image_width])

    # Перенос изображения, чтобы измерение времени соответствовало ширине изображения.
    image: Tensor = tf.transpose(a=image, perm=[1, 0, 2])
    image: Tensor = tf.expand_dims(input=image, axis=0)

    self.__logger.debug("The image is transformed.")

    return image

  @lru_cache(maxsize=EnvUtility().get_int(var_name="CORTEX_CORE_CACHE_SIZE", default_value=50))
  @traced
  def inference(self, query: str) -> str:
    """ Обрабатывает картинку с капчей и возвращает распознанные символы.

    :param query: Закодированное в Base64 изображение.

    :return: Распознанный текст с картинки.
    """

    if not EnvUtility().get_bool(var_name="CORTEX_CORE_CCS_LOG_VERBOSE_INPUT") and len(query) > 150:
      self.__logger.debug(f"Input (reduced): {query[:10]}…{query[-10:]}")
    else:
      self.__logger.debug(f"Raw input: \"{query}\".")

    # Конвертация запроса в web-safe Base64 строку.
    # Нужно, так как TensorFlow принимает такой формат.
    converted_query: str = query.replace("+", "-").replace("/", "_")

    image: Tensor = self.__transform_image(source=converted_query)

    predictions: Any = self.__model.predict(x=image)

    result: str = self.__decode_batch_predictions(predictions=predictions)

    return result
