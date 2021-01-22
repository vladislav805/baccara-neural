from typing import Tuple, List

import pandas as pd


def normalize_dataset(dataset: pd.DataFrame, mean=None, sd=None) -> Tuple[pd.DataFrame, List[float], List[float]]:
    if mean is None:
        # Среднее значение
        mean = dataset.mean(axis=0)

    if sd is None:
        # Стандартное отклонение
        sd = dataset.std(axis=0)

    # Нормализация данных
    dataset = (dataset - mean) / sd

    return dataset, mean, sd


def normalize_value(value: float, mean: float, sd: float) -> float:
    return value * sd + mean
