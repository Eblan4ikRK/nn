#!/usr/bin/env python3

""" Компонент для вывода данных (CDR).

Сделано на основе https://github.com/bartosz-paternoga/Chatbot (Bartosz Paternoga, MIT license).
"""

from functools import lru_cache
from typing import Any

from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from numpy import ndarray, argmax, intp

from runtime.core.segments.inferencer_base import Inferencer
from runtime.core.utilities.model_utils import ModelUtils
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.lib.utilities.data_utility import DataUtility
from runtime.lib.utilities.env_utility import EnvUtility


class DialogueReplyingInferencer(Inferencer):
  """ Класс для вывода данных.
  """

  @traced
  def __init__(self, model_path: str) -> None:
    """ Функция инициализации класса для вывода данных.

    :param model_path: Путь к директории модели.
    """

    super().__init__()

    self.__logger: Logger = Logger("Core/DialogueReplying/Inferencer")

    self.__logger.debug("Initializing the CDR inferencer instance…")

    self.__model: Sequential = ModelUtils().load_model(model_path=model_path)

    # region ЗАГРУЗКА МЕТАДАННЫХ.

    metadata: Any = DataUtility().load_data(filename=f"{model_path}/metadata.pkl")

    self.__tokenizer: Tokenizer = metadata["tokenizer"]
    self.__max_length: int = metadata["max_length"]

    # endregion

  @traced
  def __word_for_id(self, integer: ndarray) -> str | None:
    """ Сопоставляет число из массива со словом и возвращает слово из последовательности.

    :param integer: Число в массиве.

    :return: Слово из последовательности слов.
    """

    for word, index in self.__tokenizer.word_index.items():
      if index == integer:
        return word

    return None

  @traced
  def __predict_sequence(self, source: ndarray) -> str:
    """ Генерирует целевую последовательность слов исходя из исходных данных.

    :param source: Исходные данные в виде N-мерного массива.

    :return: Целевая последовательность слов в виде строки.
    """

    prediction: ndarray = self.__model.predict(x=source, verbose=str(0))[0]
    integers: list[ndarray] | list[intp] = [argmax(vector) for vector in prediction]
    target: list[str] = list()

    for item in integers:
      word = self.__word_for_id(integer=item)

      if word is None:
        break

      target.append(word)

    return " ".join(target)

  @traced
  def __predict(self, sources: ndarray) -> str:
    """ Получает ответ на запрос.

    :param sources: N-мерный массив данных.
    """

    self.__logger.debug("Prediction…")

    predicted: list = list()
    y_output: str = ""

    # Перебор, перевод закодированного исходного текста и обновление значения переменной ответа.
    for item, source in enumerate(sources):
      source: ndarray = source.reshape((1, source.shape[0]))
      y_output: str = self.__predict_sequence(source)

      predicted.append(y_output.split())

    self.__logger.debug(f"Prediction completed. Output: \"{y_output}\".")

    return y_output

  @lru_cache(maxsize=EnvUtility().get_int(var_name="CORTEX_CORE_CACHE_SIZE", default_value=50))
  @traced
  def inference(self, query: str) -> str:
    """ Функция обработки и вывода данных.

    :param query: Текст запроса.

    :return: Ответ.
    """

    self.__logger.debug(f"Input: \"{query.strip()}\".")

    x_input: ndarray = pad_sequences(sequences=self.__tokenizer.texts_to_sequences(query.strip().split("\n")), maxlen=self.__max_length, padding="post")

    return self.__predict(x_input)
