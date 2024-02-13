#!/usr/bin/env python3

""" Схема ответа API CDR.
"""

from pydantic import BaseModel


# Схема должна соответствовать возвращаемому значению ../objects/response.py:DialogueReplyingResponse -> to_dict().


class DialogueReplyingResponseScheme(BaseModel):
  """ Схема ответа API CDR.
  """

  # Строка ответа.
  response: str
