#!/usr/bin/env python3

""" Компонент для обучения нейросети (CCS).

Сделано на основе https://keras.io/examples/vision/captcha_ocr.
"""

import os
from typing import Any

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.engine.keras_tensor import KerasTensor
from keras.layers import StringLookup, Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.python.keras.saving.save import load_model
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.python.client import device_lib
from tensorflow.python.ops.gen_dataset_ops import TensorSliceDataset, PrefetchDataset

from runtime.core.segments.impl.captcha_solving.layers.ctc_layer import CTCLayer
from runtime.core.segments.trainer_base import Trainer
from runtime.core.utilities.model_utils import ModelUtils
from runtime.lib.logging.decorators import traced
from runtime.lib.logging.logger import Logger
from runtime.lib.utilities.data_utility import DataUtility


class CaptchaSolvingTrainer(Trainer):
  """ Класс для обучения моделей CCS.
  """

  @traced
  def __init__(self, model_path: str, dataset_path: str, model_name: str, image_width: int = 200, image_height: int = 50) -> None:
    """ Инициализирует класс.

    :param model_path: Путь к директории с моделью.
    :param dataset_path: Путь к директории с изображениями.
    :param model_name: Имя обучаемой модели.
    :param image_width: Ширина изображений.
    :param image_height: Высота изображений.
    """

    super().__init__()

    self.__logger: Logger = Logger("Core/CaptchaSolving/Trainer")

    self.__logger.debug("Initializing the CСS trainer instance…")

    self.__logger.debug("Devices for training:")

    for device in device_lib.list_local_devices():
      self.__logger.debug(f"- Device: \"{device.name}\" (type: {device.device_type}).")  # type: ignore

    if not os.path.exists(path=model_path):
      self.__logger.warning(f"The model directory \"{model_path}\" was not found.")
      self.__logger.warning("Creating a directory…")

      os.mkdir(path=model_path)

      self.__logger.warning("The directory has been created.")

    self.__model_path: str = model_path
    self.__dataset_path: str = dataset_path
    self.__model_name: str = model_name
    self.__image_width: int = image_width
    self.__image_height: int = image_height

    labeled_images: Any = self.__read_images()

    self.__max_label_length: int = labeled_images["max_label_length"]

    self.__images: ndarray = np.array(labeled_images["images"])
    self.__labels: ndarray = np.array(labeled_images["labels"])

    self.__characters: list[str] = self.__get_characters()

    self.__char_to_num: StringLookup = StringLookup(vocabulary=list(self.__characters))

  @traced
  def __read_images(self) -> dict[str, list[str] | int]:
    """ Читает изображения из директории указанной при инициализации класса.

    :return: Метаданные: Максимальная длина подписи, список изображений, список подписей.
    """

    self.__logger.info(f"Search for images in the catalog \"{self.__dataset_path}\"…")

    images: list[str] = []

    # Рекурсивное чтение директории с картинками.
    for root, subdirs, files in os.walk(top=self.__dataset_path):
      for file in files:
        if file.endswith(".png"):
          images.append(f"{root}/{file}")

    if len(images) < 1:
      self.__logger.critical(f"Directory \"{self.__dataset_path}\" does not contain images.")

      raise FileNotFoundError("Image files not found.")

    self.__logger.info(f"{len(images)} images found.")

    images: list[str] = sorted(map(str, images))

    raw_labels: list[str] = [image.split(os.path.sep)[-1].split(".png")[0] for image in images]

    max_label_length: int = max([len(label) for label in raw_labels])

    # Дополнение меток до максимального количества символов.
    labels: list[str] = [label.ljust(max_label_length) for label in raw_labels]

    return {
      "max_label_length": max_label_length,
      "images": images,
      "labels": labels
    }

  @traced
  def __get_characters(self) -> list[str]:
    """ Получает список всех возможных символов в подписи.

    :return: Список возможных символов подписи.
    """

    characters_set: set[str] = set(char for label in self.__labels for char in label)
    characters: list[str] = sorted(list(characters_set))

    return characters

  @traced
  def __split_data(self, train_size_multiplier: float, shuffle: bool = True) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Разделение данных на обучающие и проверочные наборы.

    :param train_size_multiplier: Множитель размера данных для обучения.
    :param shuffle: Будет ли перемешан массив индексов.

    :return: Кортеж из данных для обучения и данных для валидации (проверки).
    """

    # Получение общего размера датасета.
    size: int = len(self.__images)

    # Создание массива индексов.
    indices: ndarray = np.arange(size)

    # Перемешивание массива индексов, при необходимости.
    if shuffle:
      np.random.shuffle(indices)

    self.__logger.info("Splitting the dataset (train/validation).")

    # Получение размера примеров для обучения.
    train_samples: int = int(size * train_size_multiplier)

    # 4. Разделение данных на наборы для обучения и наборы для валидации.
    x_train, y_train = self.__images[indices[:train_samples]], self.__labels[indices[:train_samples]]
    x_valid, y_valid = self.__images[indices[train_samples:]], self.__labels[indices[train_samples:]]

    self.__logger.info(f"* Size of train data: {len(x_train)} entries.")
    self.__logger.info(f"* Size of validation data: {len(x_valid)} entries.")

    return x_train, x_valid, y_train, y_valid

  @tf.autograph.experimental.do_not_convert
  @traced
  def __encode_single_sample(self, image_path: str, label: str) -> dict[str, Tensor]:
    """ Кодирует пример для обучения.

    :param image_path: Путь к изображению.
    :param label: Метка изображения.

    :return: Закодированный пример для обучения.
    """

    # Чтение изображения.
    image: Tensor = tf.io.read_file(filename=image_path)

    # Декодировка и конвертация в серые цвета.
    image: Tensor = tf.io.decode_png(contents=image, channels=1)

    # Преобразование в float32 в диапазоне [0, 1].
    image: Tensor = tf.image.convert_image_dtype(image=image, dtype=tf.float32)

    # Изменение до нужного размера.
    image: Tensor = tf.image.resize(images=image, size=[self.__image_height, self.__image_width])

    # Перенос изображения, чтобы измерение времени соответствовало ширине изображения.
    image: Tensor = tf.transpose(a=image, perm=[1, 0, 2])

    # Сопоставление символов в метке с цифрами.
    new_label: Any = self.__char_to_num(tf.strings.unicode_split(input=label, input_encoding="UTF-8"))

    return {"image": image, "label": new_label}

  @traced
  def __build_model(self) -> Model:
    """ Строит и компилирует модель нейросети.

    :return: Скомпилированная модель.
    """

    input_image_layer: KerasTensor | list[Any] = Input(shape=(self.__image_width, self.__image_height, 1), name="image", dtype="float32")

    # region CONV-БЛОК №1.

    x: KerasTensor | None = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv1")(input_image_layer)
    x: KerasTensor | None = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # endregion

    # region CONV-БЛОК №2.

    x: KerasTensor | None = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2")(x)
    x: KerasTensor | None = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # endregion

    last_layer_filters: int = 64
    downsaple: int = 4  # 2 пула с шагом 2.

    new_shape: tuple[int, int] = ((self.__image_width // downsaple), (self.__image_height // downsaple) * last_layer_filters)

    x: KerasTensor | None = Reshape(target_shape=new_shape, name="reshape")(x)
    x: KerasTensor | None = Dense(units=64, activation="relu", name="dense1")(x)
    x: KerasTensor | None = Dropout(rate=0.2)(x)

    # region RNN-СЛОИ.

    x: KerasTensor | None = Bidirectional(layer=LSTM(units=128, return_sequences=True, dropout=0.25))(x)
    x: KerasTensor | None = Bidirectional(layer=LSTM(units=64, return_sequences=True, dropout=0.25))(x)

    # endregion

    x: KerasTensor | None = Dense(units=len(self.__char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    label_layer: KerasTensor | list[Any] = Input(name="label", shape=(None,), dtype="float32")

    output_layer: KerasTensor | None = CTCLayer(name="ctc_loss")(None, label_layer, x)

    model: Model = Model(inputs=[input_image_layer, label_layer], outputs=output_layer, name=self.__model_name)

    return model

  @traced
  def train(
    self,
    batch_size: int = 16,
    epochs_count: int = 100,
    overtrain: bool = False,
    early_stopping_patience: int = 10,
    train_size_multiplier: float = 0.9,
    shuffle: bool = True
  ) -> None:
    """ Обучает модель нейросети.

    :param batch_size: Размер партии примеров для обновления весов.
    :param epochs_count: Кол-во эпох.
    :param overtrain: Будет ли дообучена существующая модель, вместо обучения новой (бета).
    :param early_stopping_patience: Кол-во эпох, которое допустимо отсутствие улучшение параметров (после этого кол-ва обучение будет остановлено).
    :param train_size_multiplier: Множитель размера данных для обучения.
    :param shuffle: Перемешивание данных каждую эпоху.
    """

    x_train, x_valid, y_train, y_valid = self.__split_data(train_size_multiplier=train_size_multiplier)

    self.__logger.debug("Prefetching of a dataset…")

    # При батче > 1 изображения должны быть аналогичными (одинаковый размер и кол-во символов).

    train_dataset: TensorSliceDataset = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train))
    train_dataset: PrefetchDataset = (
      train_dataset
      .map(map_func=self.__encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size=batch_size)
      .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset: TensorSliceDataset = tf.data.Dataset.from_tensor_slices(tensors=(x_valid, y_valid))
    validation_dataset: PrefetchDataset = (
      validation_dataset
      .map(map_func=self.__encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size=batch_size)
      .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    self.__logger.debug("Dataset is prefetched.")

    self.__logger.info("Preparation of the model…")

    # Загрузка существующей модели (при включенном дообучении).
    if overtrain and os.path.exists(path=f"{self.__model_path}/trained"):
      self.__logger.info(f"-> Loading a model from \"{self.__model_path}\"…")

      model: Model | None = load_model(filepath=f"{self.__model_path}/trained", compile=False)
    else:
      self.__logger.info("-> Build new model…")

      model: Model | None = self.__build_model()

    assert model is not None

    self.__logger.info("The model is loaded.")

    model.compile(optimizer=Adam())

    self.__logger.info(f"Running \"{model.name}\" model training (overtrain: {overtrain})…")
    self.__logger.info(f"* Epochs count: {epochs_count}.")
    self.__logger.info(f"* Batch size: {batch_size}.")
    self.__logger.info(f"* Characters: {self.__characters}.")
    self.__logger.info(f"* Shuffle: {shuffle}")

    # region ФУНКЦИИ РАННЕЙ ОСТАНОВКИ.

    val_loss_monitor: EarlyStopping = EarlyStopping(
      patience=early_stopping_patience,
      restore_best_weights=True
    )

    loss_monitor: EarlyStopping = EarlyStopping(
      monitor="loss",
      patience=early_stopping_patience,
      restore_best_weights=True
    )

    model.fit(
      x=train_dataset,
      validation_data=validation_dataset,
      epochs=epochs_count,
      callbacks=[val_loss_monitor, loss_monitor],
      shuffle=shuffle
    )

    # endregion

    ModelUtils().save_model(model_path=self.__model_path, model=model)

    # TODO: Добавить вывод статистики.

    # region СОХРАНЕНИЕ МЕТА-ДАННЫХ.

    data = {
      "characters": self.__characters,
      "max_label_length": self.__max_label_length,
      "image_width": self.__image_width,
      "image_height": self.__image_height
    }

    DataUtility().save_data(filename=f"{self.__model_path}/metadata.pkl", data=data)

    # endregion

    self.__logger.info("Model training completed. The model data is saved.")
