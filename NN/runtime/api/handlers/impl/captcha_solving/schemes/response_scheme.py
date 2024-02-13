#!/usr/bin/env python3

""" Схема ответа API CCS.
"""

from pydantic import BaseModel


# Схема должна соответствовать возвращаемому значению ../objects/response.py:CaptchaSolvingResponse -> to_dict().


class CaptchaSolvingResponseScheme(BaseModel):
  """ Схема ответа API CCS.
  """

  # Строка ответа.
  response: str
