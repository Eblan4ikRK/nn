
> Комплекс нейросетей для решения различных задач, работающий на [Keras](https://keras.io/) и [TensorFlow](https://www.tensorflow.org/).

Навигация:

- Функционал
  - [Функции ядра](#ядро)
  - [Функции API](#api)
  - [Вспомогательные утилиты](#утилиты)
  - [Прочие функции](#прочее)
- [Зависимости](#зависимости)
- [Структура проекта](#структура-проекта)
- [Документация](#документация)
- [Лицензия](#лицензия)

---

## Функционал

### Ядро

- Полное логирование функций и запросов *(уровень логирования настраивается в скрипте конфигурации)*.
- LRU-кэширование ответов сегментов *(размер кэша настраивается в скрипте конфигурации)*.
- Сегменты
  - Поддержание простого диалога без контекста *(CDR)*.
  - Распознавание символов с изображения капчи *(CCS)*.

### API

- Взаимодействие с сегментами ядра через HTTP POST запросы.
- Swagger UI *(включается в скрипте конфигурации)*.

### Утилиты

- Тестирование CDR *([CDRTester](utilities/cdr_tester.sh))*.

### Прочее

- Балансировка нагрузки при работе нескольких контейнеров через Docker Compose.

---

## Зависимости

- Docker ~20.10.23.
- Docker Compose ~2.16.0.

*Зависимости Python указаны в [`requirements.txt`](requirements.txt).*

---

## Структура проекта

- Runtime
  - [API](runtime/api) - API-сервер для взаимодействия с нейросетями.
  - [Core](runtime/core) - ядро, содержащее «инференсеры».
  - [Lib](runtime/lib) - вспомогательный функционал.
- [Train](train) - скрипты для обучения моделей нейросетей сегментов.
- [Utilities](utilities) - утилиты для тестирования и отладки.

---

## Документация

- Для пользователей
  - [Установка, обучение и запуск](docs/user/startup/startup.md)
  - [Взаимодействие с API](docs/user/startup/api.md)
  - Информация о сегментах
    - [Captcha Solving *(CCS)*](docs/user/segments/captcha_solving/info.md)
    - [Dialogue Replying *(CDR)*](docs/user/segments/dialogue_replying/info.md)
- Для разработчиков
  - [Написание новых сегментов](docs/dev/write_new_segments.md)

---

## Лицензия

Copyright © 2022 [Node](https://github.com/NodesLab).

Проект распространяется под лицензией [MIT](license).