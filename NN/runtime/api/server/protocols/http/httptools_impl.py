#!/usr/bin/env python3

""" Кастомный обработчик протокола для Uvicorn (http).

Сделано на основе uvicorn/protocols/http/httptools_impl.py:HttpToolsProtocol (BSD 3-Clause License).
"""

from asyncio import Event, AbstractEventLoop, Task, Transport
from typing import Any, Optional

import httptools
from uvicorn import Config
from uvicorn.protocols.http.flow_control import service_unavailable, FlowControl
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol, RequestResponseCycle
from uvicorn.protocols.utils import get_local_addr, is_ssl, get_remote_addr
from uvicorn.server import ServerState

from runtime.api.server.protocols.http.response_cycle import CustomRequestResponseCycle
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger


class CustomHttpToolsProtocol(HttpToolsProtocol):
  """ Кастомный обработчик протокола для Uvicorn (http).
  """

  @traced
  def __init__(self, config: Config, server_state: ServerState, _loop: Optional[AbstractEventLoop] = None) -> None:
    """ Инициализирует класс.

    :param config: Конфигурация.
    :param server_state: Состояние сервера.
    :param _loop: Цикл событий.
    """

    super().__init__(
      config=config,
      server_state=server_state,
      _loop=_loop
    )

    self.__logger: Logger = Logger(name="API/Access")

  @traced
  def connection_made(self, transport: Transport) -> None:
    """ Обрабатывает установку соединения.

    :param transport: Транспорт данных.
    """

    self.connections.add(self)

    self.transport: Transport = transport
    self.flow: FlowControl = FlowControl(transport)
    self.server: tuple[str, int] | None = get_local_addr(transport)
    self.client: tuple[str, int] | None = get_remote_addr(transport)
    self.scheme: str = "https" if is_ssl(transport) else "http"

    client: tuple[str, int] | str = self.client if self.client else ""

    self.__logger.debug(f"{client[0]}:{client[1]} - HTTP connection made.")

  @traced
  def connection_lost(self, exception: Exception | None) -> None:
    """ Обрабатывает сброс соединения.

    :param exception: Ошибка (опционально).
    """

    self.connections.discard(self)

    client: tuple[str, int] | str = self.client if self.client else ""

    self.__logger.debug(f"{client[0]}:{client[1]} - HTTP connection lost.")

    if self.cycle and not self.cycle.response_complete:
      self.cycle.disconnected = True

    if self.cycle is not None:
      self.cycle.message_event.set()

    if self.flow is not None:
      self.flow.resume_writing()

    if exception is None:
      self.transport.close()

      self._unset_keepalive_if_required()

    self.parser = None

  @traced
  def data_received(self, data: bytes) -> None:
    """ Обрабатывает получение данных.

    :param data: Данные.
    """

    self._unset_keepalive_if_required()

    try:
      self.parser.feed_data(data)  # type: ignore
    except httptools.HttpParserError:
      self.__logger.warning("Invalid HTTP request received.")

      self.send_400_response("Invalid HTTP request received.")

      return
    except httptools.HttpParserUpgrade:
      upgrade: bytes | None = self._get_upgrade()

      if self._should_upgrade_to_ws(upgrade):
        self.handle_websocket_upgrade()

  @traced
  def on_headers_complete(self) -> None:
    """ Обрабатывает конец запроса.
    """

    http_version: str = self.parser.get_http_version()  # type: ignore
    method: Any = self.parser.get_method()  # type: ignore

    self.scope["method"] = method.decode("ascii")

    if http_version != "1.1":
      self.scope["http_version"] = http_version

    if self.parser.should_upgrade() and self._should_upgrade():  # type: ignore
      return

    parsed_url: Any = httptools.parse_url(self.url)  # type: ignore
    raw_path: Any = parsed_url.path
    path: Any = raw_path.decode("ascii")

    # if "%" in path:
    #   path = urllib.parse.unquote(path)

    self.scope["path"] = path
    self.scope["raw_path"] = raw_path
    self.scope["query_string"] = parsed_url.query or b""

    if self.limit_concurrency is not None and (len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency):
      app: Any = service_unavailable

      self.__logger.warning("Exceeded concurrency limit.")
    else:
      app: Any = self.app

    existing_cycle: RequestResponseCycle = self.cycle

    self.cycle: CustomRequestResponseCycle = CustomRequestResponseCycle(
      scope=self.scope,
      transport=self.transport,
      flow=self.flow,
      logger=self.logger,
      access_logger=self.access_logger,
      access_log=self.access_log,
      default_headers=self.server_state.default_headers,
      message_event=Event(),
      expect_100_continue=self.expect_100_continue,
      keep_alive=http_version != "1.0",
      on_response=self.on_response_complete,
      cortex_logger=self.__logger
    )

    if existing_cycle is None or existing_cycle.response_complete:
      task: Task = self.loop.create_task(self.cycle.run_asgi(app))
      task.add_done_callback(self.tasks.discard)

      self.tasks.add(task)
    else:
      self.flow.pause_reading()

      self.pipeline.appendleft((self.cycle, app))

  # TODO: Override run_asgi.
