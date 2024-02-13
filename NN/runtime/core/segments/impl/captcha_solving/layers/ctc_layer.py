#!/usr/bin/env python3

""" CTC-слой модели нейросети.
"""

from keras.backend import ctc_batch_cost
from keras.layers import Layer
from tensorflow import cast, shape, ones
from tensorflow.python.framework.ops import Tensor


class CTCLayer(Layer):
  """ Класс, представляющий CTC-слой нейросети.
  """

  def __init__(self, name: str | None = None) -> None:
    """ Инициализирует слой.

    :param name: Имя слоя.
    """

    super().__init__(name=name)

  def call(self, inputs: Tensor | list[Tensor] | tuple[Tensor], y_true=None, y_pred=None) -> Tensor | None:
    """ Вызывает слой.

    :param inputs: Входные тензоры (не используются).
    :param y_true: Правильный ответ.
    :param y_pred: Предсказание слоя.

    :return: Ответ.
    """

    batch_length: Tensor = cast(x=shape(y_true)[0], dtype="int64")

    input_length: Tensor = cast(x=shape(y_pred)[1], dtype="int64")
    label_length: Tensor = cast(x=shape(y_true)[1], dtype="int64")

    input_length: Tensor = input_length * ones(shape=(batch_length, 1), dtype="int64")
    label_length: Tensor = label_length * ones(shape=(batch_length, 1), dtype="int64")

    loss: Tensor = ctc_batch_cost(y_true=y_true, y_pred=y_pred, input_length=input_length, label_length=label_length)

    self.add_loss(losses=loss)

    return y_pred
