#!/usr/bin/env python3

""" Тестер для CDR.
"""

import json
from random import choice
from typing import Any

from requests import Response, post

from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.lib.utilities.string_utility import StringUtility


class CDRTester(object):
  """ Тестер для CDR.
  """

  @traced
  def __init__(self) -> None:
    """ Инициализирует класс.
    """

    self.__logger: Logger = Logger(name="CDRTester")

  @traced
  def test(self, file_path: str, cdr_url: str = "http://0.0.0.0:8080/dialogue_replying/default", tests_count: int = 100) -> None:
    """ Запускает тестирование.

    :param file_path: Путь к тестовому датасету CDR.
    :param cdr_url: URL CDR.
    :param tests_count: Количество тестов.
    """

    data_list: list[dict[str, str]] = []

    # region ЗАГРУЗКА ДАННЫХ.

    with open(file=file_path, mode="rb") as json_file:
      json_data: Any = json.load(json_file)

      for json_item in json_data:
        data_list.append({
          "query": json_item["query"],
          "response": json_item["response"]
        })

    # endregion

    correct_tests_count: int = 0
    current_tests_count: int = 0

    similarities: list[float] = []

    while current_tests_count < tests_count:
      current_tests_count += 1

      item: dict[str, str] = choice(data_list)

      response: Response = post(url=cdr_url, json={"query": item["query"]})

      prediction: str = response.json()["response"]

      similarity: float = StringUtility().get_strings_similarity(
        a=item["response"],
        b=prediction
      )

      if item["response"] == prediction:
        correct_tests_count += 1

      similarities.append(similarity)

      self.__logger.info(f"Test #{current_tests_count}")
      self.__logger.info(f"* Query: \"{item['query']}\".")
      self.__logger.info(f"* Response: \"{item['response']}\".")
      self.__logger.info(f"* Predicted: \"{prediction}\" (similarity: {similarity}%).")
      self.__logger.info()

    success_rate: float = (correct_tests_count / tests_count) * 100

    average_similarity: float = sum(similarities) / len(similarities)

    self.__logger.info("Test results:")
    self.__logger.info(f" * Correct tests count: {correct_tests_count}/{tests_count}.")
    self.__logger.info(f" * Success rate: {success_rate}%.")
    self.__logger.info(f" * Average similarity: {average_similarity}%")
