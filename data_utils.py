from typing import Optional

import numpy as np


# Создание многомерных данных
def multivariate_data(dataset,
                      target,
                      start_index: int,
                      end_index: Optional[int],
                      history_size: int,
                      target_size: int,
                      single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def pidor(dataset,
          target,
          start_index: int,
          end_index: int,
          target_size: int):
    data = []

    for i in range(256):
        data.append(dataset[:])

    return np.array(data)


def normalize_dataset(dataset, split: int):
    # Среднее значение
    mean = dataset[:split].mean(axis=0)

    # Стандартное отклонение
    sd = dataset[:split].std(axis=0)

    # Нормализация данных
    dataset = (dataset - mean) / sd

    return dataset, mean, sd


def normalize_value(value, means, sds, column):
    return value * sds[column] + means[column]
