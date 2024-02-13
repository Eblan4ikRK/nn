#!/usr/bin/env python3

""" Кастомный цикл ответов на запросы для Uvicorn (http).

Сделано на основе uvicorn/protocols/http/httptools_impl.py (BSD 3-Clause License).
"""

import logging
from asyncio import Event, Transport
from typing import Any, cast, Callable

from uvicorn.protocols.http.flow_control import CLOSE_HEADER, FlowControl
from uvicorn.protocols.http.httptools_impl import RequestResponseCycle, HEADER_RE, HEADER_VALUE_RE, STATUS_LINE
from uvicorn.protocols.utils import get_client_addr, get_path_with_query_string

from runtime.api.server.utilities.status_code_utility import StatusCodeUtility
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger


class CustomRequestResponseCycle(RequestResponseCycle):
  """ Кастомный цикл ответов на запросы для Uvicorn (http).
  """

  @traced
  def __init__(
    self,
    scope: "HTTPScope",  # type: ignore
    transport: Transport,
    flow: FlowControl,
    logger: logging.Logger,
    access_logger: logging.Logger,
    access_log: bool,
    default_headers: list[tuple[bytes, bytes]],
    message_event: Event,
    expect_100_continue: bool,
    keep_alive: bool,
    on_response: Callable[..., None],
    cortex_logger: Logger
  ):
    """ Инициализирует класс.

    :param scope: Область видимости приложения.
    :param transport: Транспорт данных.
    :param flow: Управление потоком.
    :param logger: Логер Uvicorn.
    :param access_logger: Логер доступа Uvicorn.
    :param access_log: Включение лога доступа Uvicorn.
    :param default_headers: Заголовки по умолчанию.
    :param message_event: Событие сообщения.
    :param expect_100_continue:
    :param keep_alive: Удержание соединения включенным.
    :param on_response: Функция при ответе.
    :param cortex_logger: Логер Cortex.
    """
    super().__init__(
      scope=scope,
      transport=transport,
      flow=flow,
      logger=logger,
      access_logger=access_logger,
      access_log=access_log,
      default_headers=default_headers,
      message_event=message_event,
      expect_100_continue=expect_100_continue,
      keep_alive=keep_alive,
      on_response=on_response
    )

    self.__logger: Logger = cortex_logger

  @traced
  async def run_asgi(self, app: "ASGI3Application") -> None:  # type: ignore
    """ Запускает приложения в ASGI.

    :param app: Приложение.
    """

    try:
      result = await app(self.scope, self.receive, self.send)
    except BaseException as exception:
      self.__logger.error(f"Exception in ASGI application: \"{exception}\".")

      if not self.response_started:
        await self.send_500_response()
      else:
        self.transport.close()
    else:
      if result is not None:
        self.__logger.error(f"ASGI callable should return None, but returned '{result}'.")

        self.transport.close()
      elif not self.response_started and not self.disconnected:
        self.__logger.error("ASGI callable returned without starting response.")

        await self.send_500_response()
      elif not self.response_complete and not self.disconnected:
        self.__logger.error("ASGI callable returned without completing response.")

        self.transport.close()
    finally:
      self.on_response = lambda: None

  @traced
  async def send(self, message: "ASGISendEvent") -> None:  # type: ignore
    """ Отправляет данные клиенту.

    :param message: Сообщение с данными.
    """

    message_type: Any = message["type"]

    if self.flow.write_paused and not self.disconnected:
      await self.flow.drain()

    if self.disconnected:
      return

    if not self.response_started:
      if message_type != "http.response.start":
        raise RuntimeError(f"Expected ASGI message 'http.response.start', but got '{message_type}'.")

      # message: Any = cast("HTTPResponseStartEvent", message)

      self.response_started: bool = True
      self.waiting_for_100_continue: bool = False

      status_code: Any = message["status"]
      headers: list[tuple[bytes, bytes]] = self.default_headers + list(message.get("headers", []))

      if CLOSE_HEADER in self.scope["headers"] and CLOSE_HEADER not in headers:
        headers += [CLOSE_HEADER]

      status_code_colored: str = StatusCodeUtility().get_status_code(int(status_code))

      self.__logger.info(f"{get_client_addr(self.scope)} - \u001b[37;1m\"{self.scope['method']} {get_path_with_query_string(self.scope)} HTTP/{self.scope['http_version']}\"\u001b[0m {status_code_colored}.")

      content: list[bytes] = [STATUS_LINE[status_code]]

      for name, value in headers:
        if HEADER_RE.search(name):
          raise RuntimeError("Invalid HTTP header name.")

        if HEADER_VALUE_RE.search(value):
          raise RuntimeError("Invalid HTTP header value.")

        name: bytes = name.lower()

        if name == b"content-length" and self.chunked_encoding is None:
          self.expected_content_length = int(value.decode())
          self.chunked_encoding = False
        elif name == b"transfer-encoding" and value.lower() == b"chunked":
          self.expected_content_length = 0
          self.chunked_encoding = True
        elif name == b"connection" and value.lower() == b"close":
          self.keep_alive = False

        content.extend([name, b": ", value, b"\r\n"])

      if self.chunked_encoding is None and self.scope["method"] != "HEAD" and status_code not in (204, 304):
        self.chunked_encoding = True

        content.append(b"transfer-encoding: chunked\r\n")

      content.append(b"\r\n")

      self.transport.write(b"".join(content))

    elif not self.response_complete:
      if message_type != "http.response.body":
        raise RuntimeError(f"Expected ASGI message 'http.response.body', but got '{message_type}'.")

      body: bytes = cast(bytes, message.get("body", b""))

      more_body: Any = message.get("more_body", False)

      if self.scope["method"] == "HEAD":
        self.expected_content_length: int = 0
      elif self.chunked_encoding:
        if body:
          content: list[bytes] = [b"%x\r\n" % len(body), body, b"\r\n"]
        else:
          content: list[bytes] = []
        if not more_body:
          content.append(b"0\r\n\r\n")

        self.transport.write(b"".join(content))
      else:
        num_bytes: int = len(body)

        if num_bytes > self.expected_content_length:
          raise RuntimeError("Response content longer than Content-Length.")
        else:
          self.expected_content_length -= num_bytes

        self.transport.write(body)

      if not more_body:
        if self.expected_content_length != 0:
          raise RuntimeError("Response content shorter than Content-Length.")

        self.response_complete: bool = True

        self.message_event.set()

        if not self.keep_alive:
          self.transport.close()

        self.on_response()

    else:
      raise RuntimeError(f"Unexpected ASGI message '{message_type}' sent, after response already completed.")
