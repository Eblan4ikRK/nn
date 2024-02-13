#!/usr/bin/env python3

""" Утилита для автоматического тестирования работы CCS через CLI.
"""

import base64
import os
import random
from argparse import ArgumentParser, Namespace
from difflib import SequenceMatcher

from requests import Response, post


# TODO: Переписать.


def get_strings_similarity(a: str, b: str) -> float:
  """ Получает процент схожесть строк (дистанция Левенштейна).

  :param a: Первая строка.
  :param b: Вторая строка.

  :return: Значение схожести (%).
  """

  sequence_matcher: SequenceMatcher = SequenceMatcher(a=a, b=b)

  return sequence_matcher.ratio() * 100


def get_files(images_path: str) -> tuple[list[str], dict[str, int]]:
  """ Получает список PNG файлов и словарь корневых директорий с нулевыми очками ошибок.

  :param images_path: Путь к директории с файлами.
  
  :return: Список PNG файлов и словарь корневых директорий с нулевыми очками ошибок.
  """

  images: list[str] = []
  errors: dict[str, int] = {}

  for root_dir, subdirs, files in os.walk(top=images_path):
    for file in files:
      if file.endswith(".png"):
        images.append(f"{root_dir}/{file}")

        errors[root_dir] = 0

  return images, errors


if __name__ == "__main__":
  argument_parser: ArgumentParser = ArgumentParser()

  # argument_parser.add_argument("-mp", "--model-path", type=str, required=True, help="the path to the model")
  argument_parser.add_argument("-u", "--url", type=str, default="http://0.0.0.0:8080/captcha_solving/default", help="Cortex API URL")
  argument_parser.add_argument("-ip", "--images-path", type=str, required=True, help="the path to the images directory for testing")
  argument_parser.add_argument("-tc", "--tests-count", type=int, default=100, help="number of tests")

  arguments: Namespace = argument_parser.parse_args()

  # resolver: CaptchaSolvingInferencer = CaptchaSolvingInferencer(model_path=arguments.model_path)

  files_tuple: tuple[list[str], dict[str, int]] = get_files(arguments.images_path)
  images_list: list[str] = files_tuple[0]
  error_points: dict[str, int] = files_tuple[1]

  # region Цветовые коды.

  white_color_code: str = "\u001b[37;1m"
  gray_color_code: str = "\u001b[37;2m"
  reset_color_code: str = "\u001b[0m"
  green_color_code: str = "\u001b[32m"
  yellow_color_code: str = "\u001b[33m"
  red_color_code: str = "\u001b[31m"

  # endregion

  max_tests_count: int = arguments.tests_count  # Максимальное количество тестов.
  tests_count: int = 0  # Количество пройденных тестов.
  correct_tests_count: int = 0  # Количество корректных тестов.

  while tests_count < max_tests_count:
    tests_count += 1

    # TODO: Добавить проверку на наличие файлов.
    image_path: str = random.choice(images_list)

    images_list.remove(image_path)

    with open(image_path, "rb") as image_file:
      base64_bytes: bytes = base64.b64encode(image_file.read())
      base64_string: str = base64_bytes.decode("utf-8")  # .replace("+", "-").replace("/", "_") # Конвертация байтов в web-safe Base64 строку.

    correct_text: str = image_path.split("/")[-1].replace(".png", "")

    # region Сообщение о номере теста и файле.

    colored_image_filename: str = f"{reset_color_code}{white_color_code}{correct_text}{reset_color_code}{gray_color_code}"
    colored_path: str = f"{gray_color_code}{image_path.replace(correct_text, colored_image_filename)}{reset_color_code}"

    if tests_count == 1:
      warming_up: str = "(warming up)"
    else:
      warming_up: str = ""

    print(f"Test #{tests_count}: {colored_path} {warming_up}.")

    # endregion

    response: Response = post(url=arguments.url, json={"query": base64_string})

    prediction = response.json()["response"]

    if prediction == correct_text:
      result_color_code: str = green_color_code

      correct_tests_count += 1
    else:
      result_color_code: str = red_color_code

      error_points["/".join(image_path.split("/")[: -1])] += 1

    similarity: float = get_strings_similarity(a=prediction, b=correct_text)

    print(f"Prediction result: {result_color_code}{prediction}{reset_color_code} (similarity: {similarity}%).")
    print("")

  else:
    success_rate: float = (correct_tests_count / max_tests_count) * 100

    if success_rate >= 90:
      rate_color_code: str = green_color_code
    elif success_rate >= 60:
      rate_color_code: str = yellow_color_code
    else:
      rate_color_code: str = red_color_code

    print("Test results:")
    print(f" * Correct tests: {correct_tests_count}/{tests_count}.")
    print(f" * Success rate: {rate_color_code}{success_rate}%{reset_color_code}.")
    print(" * Errors count:")

    for root in sorted(error_points.items(), key=lambda pair: pair[1], reverse=True):
      print(f"   - {gray_color_code}{root[0]}{reset_color_code}: {white_color_code}{root[1]}{reset_color_code}.")
