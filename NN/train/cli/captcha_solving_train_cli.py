#!/usr/bin/env python3

""" Скрипт для запуска обучения нейросети (CCS).
"""

from argparse import Namespace, ArgumentParser

from runtime.core.segments.impl.captcha_solving.trainer import CaptchaSolvingTrainer


class TrainerArgumentsNamespace(Namespace):
  """ Пространство имен аргументов компонента для обучения.
  """

  # Аргументы инициализации.
  model_path: str
  dataset_file: str
  model_name: str
  image_height: int
  image_width: int

  # Аргументы обучения.
  batch_size: int
  epochs_count: int
  overtrain: bool
  train_size_multiplier: float
  early_stopping_patience: int
  shuffle: bool


if __name__ == "__main__":
  argument_parser: ArgumentParser = ArgumentParser()

  # Аргументы инициализации.
  argument_parser.add_argument("-mp", "--model-path", type=str, required=True, help="the path to the model")
  argument_parser.add_argument("-dp", "--dataset-path", type=str, required=True, help="the path to the dataset directory")
  argument_parser.add_argument("-mn", "--model-name", type=str, required=True, help="name of the neural network model (ignored if overtrain is enabled)")
  argument_parser.add_argument("-ih", "--image-height", type=int, default=50, help="the height of images in the dataset")
  argument_parser.add_argument("-iw", "--image-width", type=int, default=200, help="the width of images in the dataset")

  # Аргументы обучения.
  argument_parser.add_argument("-bs", "--batch-size", type=int, default=16, help="batch size")
  argument_parser.add_argument("-ec", "--epochs-count", type=int, default=350, help="number of epochs")
  argument_parser.add_argument("-o", "--overtrain", type=bool, default=False, help="further training of the existing model")
  argument_parser.add_argument("-esp", "--early-stopping-patience", type=int, default=10, help="the number of epochs that the absence of parameter improvement is acceptable")
  argument_parser.add_argument("-dsm", "--dataset-size-multiplier", type=float, default=0.7, help="multiplier of the size of the training data")
  argument_parser.add_argument("-s", "--shuffle", type=bool, default=False, help="shuffling data every epoch")

  arguments: TrainerArgumentsNamespace | Namespace = argument_parser.parse_args()

  trainer: CaptchaSolvingTrainer = CaptchaSolvingTrainer(
    model_path=arguments.model_path,
    dataset_path=arguments.dataset_path,
    model_name=arguments.model_name,
    image_height=arguments.image_height,
    image_width=arguments.image_width
  )

  trainer.train(
    batch_size=arguments.batch_size,
    epochs_count=arguments.epochs_count,
    overtrain=arguments.overtrain,
    early_stopping_patience=arguments.early_stopping_patience,
    train_size_multiplier=arguments.dataset_size_multiplier,
    shuffle=arguments.shuffle
  )
