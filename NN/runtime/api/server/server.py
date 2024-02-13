#!/usr/bin/env python3

""" Кастомный Uvicorn сервер.

Сделано на основе uvicorn/server.py (BSD 3-Clause License).
"""

import asyncio
import os
import platform
from _socket import SocketType
from asyncio import AbstractEventLoop
from functools import partial
from socket import socket, fromfd, AF_UNIX, SOCK_STREAM
from typing import Sequence, Any

from uvicorn import Server, Config

from runtime.api.server.lifespan.on import CustomLifespanOn
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger


class CustomServer(Server):
  """ Кастомный Uvicorn сервер.
  """

  @traced
  def __init__(self, config: Config) -> None:
    super().__init__(config=config)

    self.servers: list[asyncio.base_events.Server] = []

    self.config: Config = config
    self.lifespan: Any = None

    self.__logger: Logger = Logger(name="API/Server")

  @traced
  def _log_started_message(self, listeners: Sequence[SocketType]) -> None:
    """ Записывает в лог сообщение о запуске Uvicorn.

    :param listeners: Слушатели сокетов.
    """

    if self.config.fd is not None:
      sock: SocketType = listeners[0]

      self.__logger.info(f"Uvicorn (custom) running on socket \u001b[37;1m{sock.getsockname()}\u001b[0m (Press CTRL+C to quit)")

    elif self.config.uds is not None:
      self.__logger.info(f"Uvicorn (custom) running on unix socket \u001b[37;1m{self.config.uds}\u001b[0m (Press CTRL+C to quit).")

    else:
      host: str = "0.0.0.0" if self.config.host is None else self.config.host
      port: int = self.config.port

      if port == 0:
        port: int = listeners[0].getsockname()[1]

      protocol_name: str = "https" if self.config.ssl else "http"

      self.__logger.info(f"Uvicorn (custom) running on \u001b[37;1m{protocol_name}://{host}:{port}\u001b[0m (Press CTRL+C to quit).")

  @traced
  async def serve(self, sockets: list[socket]) -> None:
    """ Запускает процесс сервера.

    :param sockets: Сокеты.
    """

    if not self.config.loaded:
      self.config.load()

    self.lifespan = CustomLifespanOn(config=self.config)

    self.install_signal_handlers()

    self.__logger.info("Started server process.")

    await self.startup(sockets=sockets)

    if self.should_exit:
      return

    await self.main_loop()
    await self.shutdown(sockets=sockets)

    self.__logger.info("Finished server process.")

  @traced
  async def startup(self, sockets: list | None = None) -> None:
    """ Запускает сервер.

    :param sockets: Список сокетов.
    """

    await self.lifespan.startup()

    if self.lifespan.should_exit:
      self.should_exit: bool = True

      return

    config: Config = self.config

    create_protocol: partial = partial(
      config.http_protocol_class,
      config=config,
      server_state=self.server_state
    )

    loop: AbstractEventLoop = asyncio.get_running_loop()

    listeners: Sequence[SocketType]

    if sockets is not None:
      def _share_socket(socket_type: SocketType) -> SocketType:
        from socket import fromshare  # type: ignore

        sock_data: bytes = socket_type.share(os.getpid())  # type: ignore

        return fromshare(sock_data)

      self.servers: list[asyncio.base_events.Server] = []

      for sock in sockets:
        if config.workers > 1 and platform.system() == "Windows":
          sock: SocketType = _share_socket(sock)

        server: asyncio.base_events.Server = await loop.create_server(
          create_protocol,
          sock=sock,  # type: ignore
          ssl=config.ssl,
          backlog=config.backlog
        )

        self.servers.append(server)

      listeners: Sequence[SocketType] = sockets

    elif config.fd is not None:
      sock: SocketType = fromfd(config.fd, AF_UNIX, SOCK_STREAM)

      server: asyncio.base_events.Server = await loop.create_server(
        create_protocol,
        sock=sock,
        ssl=config.ssl,
        backlog=config.backlog
      )

      assert server.sockets is not None

      listeners: Sequence[SocketType] = server.sockets

      self.servers: list[asyncio.base_events.Server] = [server]

    elif config.uds is not None:
      uds_perms: int = 0o666

      if os.path.exists(config.uds):
        uds_perms: int = os.stat(config.uds).st_mode

      server: asyncio.base_events.Server = await loop.create_unix_server(
        create_protocol,
        path=config.uds,
        ssl=config.ssl,
        backlog=config.backlog
      )

      os.chmod(config.uds, uds_perms)

      assert server.sockets is not None

      listeners: Sequence[SocketType] = server.sockets

      self.servers: list[asyncio.base_events.Server] = [server]

    else:
      try:
        server: asyncio.base_events.Server = await loop.create_server(
          create_protocol,
          host=config.host,
          port=config.port,
          ssl=config.ssl,
          backlog=config.backlog,
        )
      except OSError as exception:
        self.__logger.error(f"An error occurred when starting the server: \"{exception}\".")

        await self.lifespan.shutdown()

        exit(1)

      assert server.sockets is not None

      listeners: Sequence[SocketType] = server.sockets

      self.servers: list[asyncio.base_events.Server] = [server]

    if sockets is None:
      self._log_started_message(listeners=listeners)
    else:
      pass

    self.started: bool = True

  @traced
  async def shutdown(self, sockets: list[socket] | None = None) -> None:
    """ Останавливает сервер.

    :param sockets: Сокеты.
    """

    self.__logger.info("Shutting down…")

    # Остановка приема новых соединений.
    for server in self.servers:
      server.close()
    for sock in sockets or []:
      sock.close()
    for server in self.servers:
      await server.wait_closed()

    # Обрыв всех существующих соединений.
    for connection in list(self.server_state.connections):
      connection.shutdown()

    await asyncio.sleep(delay=0.1)

    # Ожидание пока существующие соединения закончат отправку ответов.
    if self.server_state.connections and not self.force_exit:
      self.__logger.info("Waiting for connections to close. (CTRL+C to force quit)…")

      while self.server_state.connections and not self.force_exit:
        await asyncio.sleep(delay=0.1)

    # Ожидание завершения текущих задач.
    if self.server_state.tasks and not self.force_exit:
      self.__logger.info("Waiting for background tasks to complete. (CTRL+C to force quit)…")

      while self.server_state.tasks and not self.force_exit:
        await asyncio.sleep(0.1)

    # Отправка события завершения работы и ожидание завершения работы приложения.
    if not self.force_exit:
      await self.lifespan.shutdown()
