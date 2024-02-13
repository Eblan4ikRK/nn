#!/usr/bin/env python3

""" Статичный логер.
"""

from runtime.lib.logging.logger import Logger
from runtime.lib.decorators import singleton


@singleton
class StaticLogger(Logger):
  """ Статичная обёртка над логером в виде синглтона.
  """
