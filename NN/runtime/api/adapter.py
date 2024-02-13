#!/usr/bin/env python3

""" Адаптер для соединения сервера Uvicorn с приложением FastAPI.
"""

from typing import BinaryIO

import yaml
from fastapi import FastAPI, APIRouter

from runtime.api.handlers.handler_base import Handler
from runtime.api.handlers.handler_manager import HandlerManager
from runtime.api.server.config import CustomConfig
from runtime.api.server.server import CustomServer
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.meta import version, generation


class Adapter(object):
  """ Адаптер для соединения сервера Uvicorn с приложением FastAPI.
  """

  @traced
  def __init__(self, enable_swagger: bool = False) -> None:
    """ Инициализирует адаптер.

    :param enable_swagger: Включение документации Swagger.
    """

    self.__logger: Logger = Logger(name=f"API/Adapter")

    self.__logger.debug(f"Инициализирую проверку адаптера AI")

    swagger_url: str | None = "/swagger" if enable_swagger else None

    self.app: FastAPI = FastAPI(
      title="CortexAPI",
      version=version,
      description="Cortex API",
      docs_url=swagger_url,
      redoc_url=None
    )

  @traced
  def __include_router(self, router: APIRouter) -> None:
    """ Подключает роутер FastAPI к приложению.

    :param router: Экземпляр роутера FastAPI.
    """

    self.__logger.debug("Including router to server app…")

    self.app.include_router(router=router)

    self.__logger.debug("The router is included.")

  @traced
  def __handle_routers(self, routers_config: dict, router_handler: Handler) -> None:
    """ Обрабатывает роутеры сегмента.

    Создает для каждой записи из конфигурации роутер FastAPI.

    :param routers_config: Конфигурация роутеров.
    :param router_handler: Класс обработчика роутера.
    """

    # TODO: Сделать свой объект для конфигурации роутеров.

    for router_config in routers_config[router_handler.segment_name]["routers"]:
      router_path: str = f"/{router_config['path']}"

      self.__logger.debug(f"Initialization of the \"/{router_handler.segment_name}{router_path}\" router…")

      try:
        router: APIRouter = router_handler.create_router(
          model_path=router_config["model"],
          path=router_path,
          summary=router_config["description"]
        )
      except FileNotFoundError as exception:
        self.__logger.warning(f"The router is not initialized. Reason: {exception}")

        continue

      self.__include_router(router=router)

  @traced
  def __connect_routers(self, routers_file: str) -> None:
    """ Обрабатывает роутеры сегментов, указанные в конфигурационном файле и подключает их к FastAPI-приложению.

    :param routers_file: Файл с настройками роутеров.
    """

    try:
      routers_config_file: BinaryIO = open(file=routers_file, mode="rb")
      routers_config: dict = yaml.load(stream=routers_config_file, Loader=yaml.CLoader)

      self.__logger.debug("Routers settings are loaded, connecting routers by configuration…")

      # На каждую указанную в конфигурации запись создается роутер.
      for segment_name in routers_config:
        if not routers_config[segment_name]["enable"]:
          self.__logger.debug(f"The routing of the \"{segment_name}\" segment is disabled.")

          continue

        self.__logger.debug(f"Initialization of segment \"{segment_name}\" routers…")

        try:
          router_handler: Handler = HandlerManager().get_handler(segment_name=segment_name)

          self.__handle_routers(routers_config=routers_config, router_handler=router_handler)
        except NotImplementedError as exception:
          self.__logger.error(exception)

          continue

      self.__logger.debug("Connection of routers is completed.")
    except FileNotFoundError:
      self.__logger.error(f"The specified file \"{routers_file}\" with the configuration of routers does not exist.")

  @traced
  def __init_cortex_features(self, routers_file: str) -> None:
    """ Подключает для приложения FastAPI рабочие функции (роутеры).

    :param routers_file: Файл с настройками роутеров.
    """

    self.__logger.info("Initialization of Cortex features…")

    self.__connect_routers(routers_file=routers_file)

    self.__logger.info("Cortex features is initialized.")

  @traced
  def run_server(self, host: str = "0.0.0.0", port: int = 8000, routers_file: str = "routers.yml") -> None:
    """ Запускает сервер.

    :param host: Хост сервера.
    :param port: Порт, на котором будет запущен сервер.
    :param routers_file: Файл с настройками роутеров.
    """

    if port > 65535:
      self.__logger.critical("The port must not exceed the value 65535.")

      exit(1)

    self.__logger.info("___________   _____  .___         ")
    self.__logger.info("\_   _____/  /  _  \ |   | " + "\u001b[91m| |\u001b[0m")
    self.__logger.info(" |    __)_  /  /_\  \|   |    " + "\u001b[91m| |\u001b[0m")
    self.__logger.info(" |        \/    |    \   |  " + "\u001b[91m| |\u001b[0m")
    self.__logger.info("/_______  /\____|__  /___|  " + "\u001b[91m| |\u001b[0m")
    self.__logger.info("        \/         \/       " + "\u001b[91m| |\u001b[0m")
    self.__logger.info()
    self.__logger.info(f"Running the AI Site (v{version}, gen. {generation})…")

    self.__logger.info(f"* Launching on Uvicorn (custom).")

    self.__init_cortex_features(
      routers_file=routers_file
    )

    config: CustomConfig = CustomConfig(
      app=self.app,
      host=host,
      port=port
    )

    server: CustomServer = CustomServer(config=config)

    server.run()
