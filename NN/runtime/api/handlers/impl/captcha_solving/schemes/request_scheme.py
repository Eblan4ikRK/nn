#!/usr/bin/env python3

""" Схема запроса к API CCS.
"""

from pydantic import BaseModel


class CaptchaSolvingRequestScheme(BaseModel):
  """ Схема запроса к API CCS.
  """

  # Закодированное в Base64 изображение.
  query: str
