#!/usr/bin/env python3

""" Схема запроса к API CDR.
"""

from pydantic import BaseModel


class DialogueReplyingRequestScheme(BaseModel):
  """ Схема запроса к API CDR.
  """

  # Текст запроса.
  query: str
