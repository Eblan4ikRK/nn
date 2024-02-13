#!/usr/bin/env python3

""" Компонент для обучения нейросети (CDR).

Сделано на основе https://github.com/bartosz-paternoga/Chatbot (Bartosz Paternoga, MIT license).
"""

import json
import os
from typing import Any

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.saving.save import load_model
from keras.utils import to_categorical, pad_sequences
from numpy import array, ndarray, arange
from numpy import random
from tensorflow.python.client import device_lib

from runtime.core.segments.trainer_base import Trainer
from runtime.core.utilities.model_utils import ModelUtils
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.lib.utilities.data_utility import DataUtility


class DialogueReplyingTrainer(Trainer):
  """ Класс для обучения нейросети.
  """

  @traced
  def __init__(self, model_path: str, dataset_file: str, model_name: str) -> None:
    """ Инициализирует класс.

    :param model_path: Путь к директории с моделью.
    :param dataset_file: Файл датасета.
    :param model_name: Имя обучаемой модели.
    """

    super().__init__()

    self.__logger: Logger = Logger("Core/DialogueReplying/Trainer")

    self.__logger.debug("Initializing the CDR trainer instance…")

    self.__logger.debug("Devices for training:")

    for device in device_lib.list_local_devices():
      self.__logger.debug(f"- Device: \"{device.name}\" (type: {device.device_type}).")  # type: ignore

    if not os.path.exists(path=model_path):
      self.__logger.warning(f"The model directory \"{model_path}\" was not found.")
      self.__logger.warning("Creating a directory…")

      os.mkdir(path=model_path)

      self.__logger.warning("The directory has been created.")

    if not os.path.exists(dataset_file):
      self.__logger.critical("Dataset file is not exixts. Create and fill it with data.")

      raise FileNotFoundError("The dataset file does not exist.")

    self.__model_path: str = model_path
    self.__dataset_file: str = dataset_file
    self.__model_name: str = model_name

    self.__dataset: ndarray = self.__load_json_dataset()

  @traced
  def __load_json_dataset(self) -> ndarray:
    """ Загружает датасет из JSON-файла в N-мерный массив.

    :return: Датасет в виде N-мерного массива.
    """

    self.__logger.info(f"Loading data from \"{self.__dataset_file}\"…")

    with open(self.__dataset_file, "rb") as json_file:
      data_list: list = []
      json_data: Any = json.load(json_file)

      for json_item in json_data:
        data_list.append([json_item["query"], json_item["response"]])

      self.__logger.info(f"Data successfully loaded ({len(json_data)} entries).")

      return array(data_list)

  @traced
  def __reformat_dataset(self, data: ndarray) -> ndarray:
    """ Обрезает и перемешивает данные из датасета.

    :param data: Датасет (в виде N-мерного массива).

    :return: Обрезанный до указанного размера датасет с перемешанными данными.
    """

    self.__logger.debug("Shuffling the dataset…")

    # Перетасовка датасета в случайном порядке.
    random.shuffle(data)

    self.__logger.debug("Dataset shuffling is complete.")

    return data

  @traced
  def __split_dataset(self, train_size_multiplier: float) -> tuple[ndarray, ndarray]:
    """ Разделяет датасет на train и validation.

    :param train_size_multiplier: Множитель используемых данных.

    :return: Данные для обучения и валидации в виде кортежа (train, validation).
    """

    reformatted_dataset: ndarray = self.__reformat_dataset(data=self.__dataset)

    train_size: int = int(len(reformatted_dataset) * train_size_multiplier)
    indices: ndarray = arange(len(reformatted_dataset))

    self.__logger.info("Splitting the dataset (train/validation).")

    train_data: ndarray = reformatted_dataset[:indices[train_size]]
    validation_data: ndarray = reformatted_dataset[indices[train_size]:]

    self.__logger.info(f"* Size of train data: {len(train_data)} entries.")
    self.__logger.info(f"* Size of validation data: {len(validation_data)} entries.")

    return train_data, validation_data

  @traced
  def __create_tokenizer(self, lines: ndarray) -> Tokenizer:
    """ Создаёт токенизатор.

    :param lines: Строки в виде N-мерного массива.

    :return: Токенизатор.
    """

    self.__logger.debug("Creating a tokenizer…")

    tokenizer: Tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=lines)

    self.__logger.debug("The tokenizer has been successfully created.")

    return tokenizer

  @staticmethod
  @traced
  def __encode_sequences(tokenizer: Tokenizer, lines: ndarray, max_length: int) -> ndarray:
    """ Кодирует и заполняет последовательности.

    :param tokenizer: Токенизатор.
    :param lines: N-мерный массив строк.
    :param max_length: Максимальная длина строки.

    :return: Закодированный и заполненный N-мерный массив.
    """

    # Кодирование последовательностей.
    x_input: ndarray | list = tokenizer.texts_to_sequences(lines)

    # Заполнение последовательностей с нулевым значением.
    x_input: ndarray | list = pad_sequences(x_input, maxlen=max_length, padding="post")

    return x_input

  @staticmethod
  @traced
  def __encode_output(encoded_sequences: ndarray, vocabulary_size: int) -> ndarray:
    """ Кодирует целевые данные.

    :param encoded_sequences: N-мерный массив целевых данных.
    :param vocabulary_size: Размер словаря.

    :return: Закодированный N-мерный массив целевых данных.
    """

    y_list: list = list()

    for sequence in encoded_sequences:
      encoded: ndarray = to_categorical(y=sequence, num_classes=vocabulary_size)

      y_list.append(encoded)

    # Создание N-мерного массива и его решейп.
    y_output: ndarray = array(y_list).reshape((encoded_sequences.shape[0], encoded_sequences.shape[1], vocabulary_size))

    return y_output

  @traced
  def __build_model(self, max_length: int, vocabulary_size: int) -> Sequential:
    """ Строит модель нейросети.

    :param max_length: Максимальная длина строки.
    :param vocabulary_size: Размер словаря.

    :return: Модель нейросети.
    """

    n_units: int = 256

    model: Sequential = Sequential(name=self.__model_name)

    model.add(Embedding(vocabulary_size, n_units, input_length=max_length, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(max_length))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(vocabulary_size, activation="softmax")))

    return model

  @traced
  def train(
    self,
    epochs_count: int = 100,
    batch_size: int = 64,
    overtrain: bool = False,
    train_size_multiplier: float = 0.5,
    early_stopping_patience: int = 10,
    shuffle: bool = True
  ) -> None:
    """ Запускает обучение нейросети.

    :param epochs_count: Количество эпох.
    :param batch_size: Размер батча (партии примеров для обновления весов).
    :param overtrain: Будет ли дообучена существующая модель, вместо обучения новой (бета).
    :param train_size_multiplier: Множитель используемых данных (остальные данные будут использованы для валидации).
    :param early_stopping_patience: Кол-во эпох, которое допустимо отсутствие улучшение параметров (после этого кол-ва обучение будет остановлено).
    :param shuffle: Перемешивание данных каждую эпоху.
    """

    # Разделение переформатированного датасета на train / validation.
    train_data, validate_data = self.__split_dataset(train_size_multiplier=train_size_multiplier)

    self.__dataset: ndarray = self.__dataset.reshape(-1, 1)

    tokenizer: Tokenizer = self.__create_tokenizer(self.__dataset[:, 0])

    # Размер словаря.
    vocabulary_size: int = len(tokenizer.word_index) + 1

    # Максимальная длина строки.
    max_length: int = max(len(line.split()) for line in self.__dataset[:, 0])

    # Подготовка данных для обучения.
    train_x: ndarray = self.__encode_sequences(tokenizer=tokenizer, lines=train_data[:, 0], max_length=max_length)
    train_y: ndarray = self.__encode_output(encoded_sequences=self.__encode_sequences(tokenizer=tokenizer, lines=train_data[:, 1], max_length=max_length), vocabulary_size=vocabulary_size)

    # Подготовка данных для валидации.
    validate_x: ndarray = self.__encode_sequences(tokenizer=tokenizer, lines=validate_data[:, 0], max_length=max_length)
    validate_y: ndarray = self.__encode_output(encoded_sequences=self.__encode_sequences(tokenizer=tokenizer, lines=validate_data[:, 1], max_length=max_length), vocabulary_size=vocabulary_size)

    self.__logger.info("Preparation of the model…")

    # Загрузка существующей модели (при включенном дообучении).
    if overtrain and os.path.exists(path=f"{self.__model_path}/trained"):
      self.__logger.info(f"-> Loading a model from \"{self.__model_path}\"…")

      model: Sequential | None = load_model(filepath=f"{self.__model_path}/trained")
    else:
      self.__logger.info("-> Build new model…")

      model: Sequential | None = self.__build_model(max_length=max_length, vocabulary_size=vocabulary_size)

    assert model is not None

    self.__logger.info("The model is loaded.")

    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])

    self.__logger.info(f"Running \"{model.name}\" model training (overtrain: {overtrain})…")
    self.__logger.info(f"* Epochs count: {epochs_count}.")
    self.__logger.info(f"* Batch size: {batch_size}.")
    self.__logger.info(f"* Vocabulary Size: {vocabulary_size} words.")
    self.__logger.info(f"* Max input length: {max_length} words.")
    self.__logger.info(f"* Shuffle: {shuffle}")

    accuracy_monitor: EarlyStopping = EarlyStopping(
      monitor="accuracy",
      patience=early_stopping_patience,
      restore_best_weights=True,
      verbose=1
    )

    model.fit(
      x=train_x,
      y=train_y,
      epochs=epochs_count,
      batch_size=batch_size,
      validation_data=(validate_x, validate_y),
      callbacks=[accuracy_monitor],
      shuffle=shuffle
    )

    ModelUtils().save_model(model_path=self.__model_path, model=model)

    # TODO: Добавить вывод статистики.

    # region СОХРАНЕНИЕ МЕТА-ДАННЫХ.

    data: dict[str, Tokenizer | int] = {
      "tokenizer": tokenizer,
      "max_length": max_length
    }

    DataUtility().save_data(filename=f"{self.__model_path}/metadata.pkl", data=data)

    # endregion

    self.__logger.info("Model training completed. The model data is saved.")
