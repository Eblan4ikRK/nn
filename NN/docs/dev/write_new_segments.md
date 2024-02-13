# Написание новых сегментов Cortex

Навигация:

- [Создание классов для вывода данных и обучения](#создание-классов-для-вывода-данных-и-обучения)
- [Создание скриптов для обучения нейросетей](#создание-скриптов-для-обучения-нейросетей)
- [Создание и подключение обработчика роутеров к серверу](#создание-и-подключение-обработчика-роутеров-к-серверу)
- [Написание документации](#написание-документации)

---

## Создание классов для вывода данных и обучения

Классы для вывода данных и классы-тренеры находятся в [ядре Cortex-Runtime](../../runtime/core/segments/impl).

> [Пример класса для вывода данных](../../runtime/core/segments/impl/captcha_solving/inferencer.py)

<!--  -->

> [Пример класса-тренера](../../runtime/core/segments/impl/captcha_solving/trainer.py)

---

## Создание скриптов для обучения нейросетей

Скрипты для обучения моделей нейросетей состоят из 2 файлов:

1. `train/cli/<segment_name>_train_cli.py` - Файл с CLI-интерфейсом для работы с классом-тренером.
2. `train/<segment_name>_train.sh` - Файл для запуска обучения с прописанной конфигурацией.

> [Пример CLI-интерфейса](../../train/cli/captcha_solving_train_cli.py)

<!--  -->

> [Пример скрипта для запуска обучения](../../train/captcha_solving_train.sh)

---

## Создание и подключение обработчика роутеров к серверу

### Создание класса обработчика роутеров

Обработчик роутеров необходим для возможности создания отдельных роутеров на каждую обученную модель.

Перед подключением обработчика роутеров к серверу необходимо создать его в [API Cortex-Runtime](../../runtime/api/handlers/impl). Для его создания необходим собственно класс обработчика, наследующий [базовый](../../runtime/api/handlers/handler_base.py) объект ответа, а также схемы ответа и запроса к роутерам, которые создает этот обработчик.

> [Пример обработчика роутеров](../../runtime/api/handlers/impl/captcha_solving/handler.py)

<!--  -->

> [Пример объекта ответа](../../runtime/api/handlers/impl/captcha_solving/objects/response.py)

<!--  -->

> [Пример схемы запроса к роутеру](../../runtime/api/handlers/impl/captcha_solving/schemes/request_scheme.py)

<!--  -->

> [Пример схемы ответа роутера](../../runtime/api/handlers/impl/captcha_solving/schemes/response_scheme.py)

### Подключение обработчика роутеров к серверу

После создания обработчика роутеров его нужно подключить к [менеджеру обработчиков роутеров в API Cortex-Runtime](../../runtime/api/handlers/handler_manager.py):

```python
# ...

self.handlers: list[RouterHandler] = [
  CaptchaSolvingHandler(),
  DialogueReplyingHandler(),
  # ...
  SegmentNameHandler()  # <- Подключение обработчика роутеров.
]

# ...
```

---

## Написание документации

После написания сегмента, необходимо создать документацию, для работы с ним. Расположить созданный документ можно в [docs/user/segments](../user/segments), где нужно создать директорию с именем сегмента, внутри которой создать файл с именем `info.md`.

> [Пример документации сегмента](../user/segments/captcha_solving/info.md)