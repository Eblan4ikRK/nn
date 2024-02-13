#!/usr/bin/env python3

""" Скрипт для запуска обучения нейросети (CDR).
"""

from argparse import Namespace, ArgumentParser

from runtime.core.segments.impl.dialogue_replying.trainer import DialogueReplyingTrainer


class TrainerArgumentsNamespace(Namespace):
  """ Пространство имен аргументов компонента для обучения.
  """

  # Аргументы инициализации.
  model_path: str
  dataset_file: str
  model_name: str

  # Аргументы обучения.
  epochs_count: int
  batch_size: int
  overtrain: bool
  dataset_size_multiplier: float
  early_stopping_patience: int
  shuffle: bool


if __name__ == "__main__":
  argument_parser: ArgumentParser = ArgumentParser()

  # Аргументы инициализации.
  argument_parser.add_argument("-mp", "--model-path", type=str, required=True, help="the path to the model directory")
  argument_parser.add_argument("-df", "--dataset-file", type=str, required=True, help="the path to the dataset file")
  argument_parser.add_argument("-mn", "--model-name", type=str, required=True, help="name of the neural network model")

  argument_parser.add_argument("-ec", "--epochs-count", type=int, default=200, help="number of epochs")
  argument_parser.add_argument("-bs", "--batch-size", type=int, default=16, help="batch size")
  argument_parser.add_argument("-o", "--overtrain", type=bool, default=False, help="further training of the existing model")
  argument_parser.add_argument("-dsm", "--dataset-size-multiplier", type=float, default=0.5, help="multiplier of the data used for train")
  argument_parser.add_argument("-esp", "--early-stopping-patience", type=int, default=50, help="the number of epochs that the absence of parameter improvement is acceptable")
  argument_parser.add_argument("-s", "--shuffle", type=bool, default=False, help="shuffling data every epoch")

  arguments: TrainerArgumentsNamespace | Namespace = argument_parser.parse_args()

  trainer: DialogueReplyingTrainer = DialogueReplyingTrainer(
    model_path=arguments.model_path,
    dataset_file=arguments.dataset_file,
    model_name=arguments.model_name
  )

  trainer.train(
    epochs_count=arguments.epochs_count,
    batch_size=arguments.batch_size,
    overtrain=arguments.overtrain,
    train_size_multiplier=arguments.dataset_size_multiplier,
    early_stopping_patience=arguments.early_stopping_patience,
    shuffle=arguments.shuffle
  )
