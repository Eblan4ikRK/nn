#!/usr/bin/env python3

""" Кастомный Uvicorn конфиг.

Сделано на основе uvicorn/config.py (BSD 3-Clause License).
"""

import asyncio
import inspect
import os
import socket
import sys
from asyncio import Protocol
from socket import AddressFamily
from ssl import SSLContext
from typing import Optional, Type, Any

from fastapi import FastAPI
from uvicorn import Config
from uvicorn.config import HTTP_PROTOCOLS, create_ssl_context, WS_PROTOCOLS
from uvicorn.importer import import_from_string, ImportFromStringError
from uvicorn.middleware.asgi2 import ASGI2Middleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from uvicorn.middleware.wsgi import WSGIMiddleware

from runtime.api.server.lifespan.on import CustomLifespanOn
from runtime.api.server.protocols.http.httptools_impl import CustomHttpToolsProtocol
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger


class CustomConfig(Config):
  """ Кастомный Uvicorn конфиг.
  """

  @traced
  def __init__(self, app: str | FastAPI, host: str, port: int) -> None:
    """ Инициализирует конфиг.

    :param app: Приложение FastAPI, либо путь к его объекту.
    :param host: Хост сервера.
    :param port: Порт сервера.
    """

    super().__init__(
      app=app,
      host=host,
      port=port,
      workers=1,
      http="httptools"
    )

    self.loaded_app = None
    self.ws_protocol_class: Type[asyncio.Protocol] | None = None
    self.ssl: SSLContext | None = None
    self.http_protocol_class: Type[Protocol] | None = None
    self.lifespan_class: Type[CustomLifespanOn] | None = None

    # Установка кастомного обработчика протокола для http.
    HTTP_PROTOCOLS["httptools"] = "runtime.api.server.protocols.http.httptools_impl:CustomHttpToolsProtocol"

    self.__logger: Logger = Logger(name="API/Config")

  def bind_socket(self) -> socket.socket:
    """ Подключает сокет.

    :return: Сокет.
    """

    if self.uds:
      path: str = self.uds
      sock: socket.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

      try:
        sock.bind(path)

        uds_perms: int = 0o666

        os.chmod(self.uds, uds_perms)
      except OSError as exception:
        self.__logger.error(exception)

        sys.exit(1)
    elif self.fd:
      sock: socket.socket = socket.fromfd(self.fd, socket.AF_UNIX, socket.SOCK_STREAM)
    else:
      family: AddressFamily = socket.AF_INET

      if self.host and ":" in self.host:
        family = socket.AF_INET6

      sock: socket.socket = socket.socket(family=family)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

      try:
        sock.bind((self.host, self.port))
      except OSError as exception:
        self.__logger.error(exception)

        sys.exit(1)

    protocol_name: str = "https" if self.is_ssl else "http"

    self.__logger.info(f"Uvicorn running on \u001b[37;1m{protocol_name}://{self.host}:{self.port}\u001b[0m (Press CTRL+C to quit).")

    sock.set_inheritable(True)

    return sock

  def load(self) -> None:
    """ Загружает конфигурацию.
    """

    assert not self.loaded

    if self.is_ssl:
      assert self.ssl_certfile

      self.ssl: Optional[SSLContext] = create_ssl_context(
        keyfile=self.ssl_keyfile,
        certfile=self.ssl_certfile,
        password=self.ssl_keyfile_password,
        ssl_version=self.ssl_version,
        cert_reqs=self.ssl_cert_reqs,
        ca_certs=self.ssl_ca_certs,
        ciphers=self.ssl_ciphers,
      )
    else:
      self.ssl = None

    encoded_headers: list[tuple[bytes, bytes]] = [
      (key.lower().encode("latin1"), value.encode("latin1"))
      for key, value in self.headers
    ]

    self.encoded_headers: list[tuple[bytes, bytes]] = (
      [(b"server", b"uvicorn")] + encoded_headers
      if b"server" not in dict(encoded_headers) and self.server_header
      else encoded_headers
    )

    if isinstance(self.http, str):
      self.http_protocol_class = CustomHttpToolsProtocol
    else:
      self.http_protocol_class = self.http

    if isinstance(self.ws, str):
      ws_protocol_class: Type[Protocol] = import_from_string(WS_PROTOCOLS[self.ws])  # type: ignore

      self.ws_protocol_class: Optional[Type[Protocol]] = ws_protocol_class
    else:
      self.ws_protocol_class = self.ws

    self.lifespan_class = CustomLifespanOn

    try:
      self.loaded_app: Any = import_from_string(self.app)
    except ImportFromStringError as exception:
      self.__logger.error(f"Error loading ASGI app:\n{exception}")

      sys.exit(1)

    try:
      self.loaded_app: Any = self.loaded_app()
    except TypeError as exception:
      if self.factory:
        self.__logger.error(f"Error loading ASGI app factory: {exception}")

        sys.exit(1)
    else:
      if not self.factory:
        self.__logger.warning("ASGI app factory detected.")

    if self.interface == "auto":
      if inspect.isclass(self.loaded_app):
        use_asgi_3: bool = hasattr(self.loaded_app, "__await__")
      elif inspect.isfunction(self.loaded_app):
        use_asgi_3: bool = asyncio.iscoroutinefunction(self.loaded_app)
      else:
        call: Any = getattr(self.loaded_app, "__call__", None)

        use_asgi_3: bool = asyncio.iscoroutinefunction(call)

      self.interface: str = "asgi3" if use_asgi_3 else "asgi2"

    if self.interface == "wsgi":
      self.loaded_app: Any = WSGIMiddleware(app=self.loaded_app)

      self.ws_protocol_class: Type[Protocol] | None = None
    elif self.interface == "asgi2":
      self.loaded_app: Any = ASGI2Middleware(app=self.loaded_app)

    if self.proxy_headers:
      self.loaded_app: Any = ProxyHeadersMiddleware(app=self.loaded_app, trusted_hosts=self.forwarded_allow_ips)

    self.loaded: bool = True
