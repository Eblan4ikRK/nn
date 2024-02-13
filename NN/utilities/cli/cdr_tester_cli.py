#!/usr/bin/env python3

""" CLI для утилиты CDRTester.
"""

from argparse import Namespace, ArgumentParser

from utilities.base.cdr_tester import CDRTester


class CDRTesterArgumentsNamespace(Namespace):
  """ Пространство имен аргументов.
  """

  file_path: str
  cdr_url: str
  tests_count: int


if __name__ == "__main__":
  argument_parser: ArgumentParser = ArgumentParser()

  argument_parser.add_argument("-fp", "--file-path", type=str, required=True, help="the path to the CDR dataset")
  argument_parser.add_argument("-u", "--cdr-url", type=str, required=True, help="the CDR url")
  argument_parser.add_argument("-tc", "--tests-count", type=int, default=100, help="number of tests")

  arguments: CDRTesterArgumentsNamespace | Namespace = argument_parser.parse_args()

  cdr_tester: CDRTester = CDRTester()

  cdr_tester.test(
    file_path=arguments.file_path,
    cdr_url=arguments.cdr_url,
    tests_count=arguments.tests_count
  )
