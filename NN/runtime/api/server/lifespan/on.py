#!/usr/bin/env python3

""" Кастомный жизненный цикл для Uvicorn.

Сделано на основе uvicorn/lifespan/on.py (BSD 3-Clause License).
"""

import asyncio
from asyncio import AbstractEventLoop

from uvicorn import Config
from uvicorn.lifespan.on import LifespanOn

from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger


class CustomLifespanOn(LifespanOn):
  """ Кастомный жизненный цикл для Uvicorn.
  """

  @traced
  def __init__(self, config: Config) -> None:
    """ Инициализирует класс.

    :param config: Экземпляр конфигурации Uvicorn.
    """

    super().__init__(config=config)

    self.__logger: Logger = Logger(name="API/Lifespan")

  @traced
  async def startup(self) -> None:
    """ Запускает приложение.
    """

    self.__logger.info("Waiting for application startup…")

    loop: AbstractEventLoop = asyncio.get_event_loop()

    loop.create_task(self.main())

    startup_event: dict[str, str] = {"type": "lifespan.startup"}

    await self.receive_queue.put(startup_event)
    await self.startup_event.wait()

    if self.startup_failed or (self.error_occured and self.config.lifespan == "on"):
      self.__logger.error("Application startup failed. Exiting.")

      self.should_exit = True
    else:
      self.__logger.info("Application startup complete.")

  # TODO: Override main.

  @traced
  async def shutdown(self) -> None:
    """ Останавливает работу приложения.
    """

    if self.error_occured:
      return

    self.__logger.info("Waiting for application shutdown…")

    shutdown_event: dict[str, str] = {"type": "lifespan.shutdown"}

    await self.receive_queue.put(shutdown_event)
    await self.shutdown_event.wait()

    if self.shutdown_failed or (self.error_occured and self.config.lifespan == "on"):
      self.__logger.error("Application shutdown failed. Exiting.")

      self.should_exit = True
    else:
      self.__logger.info("Application shutdown complete.")
