import numpy as np
import pandas as pd


def split_dataset(dataset: pd.Series, by: int):
    return dataset[:by], dataset[by:]


# нет поддержки single_step = False
def multivariate_dataset(dataset,  # наш датасет (матрица)
                         column_index: int,  # какой столбец будем обучать
                         history_size: int,  # сколько строк в прошлом будем брать
                         target_size: int,  # сколько вперед считаем
                         ):
    xs = []  # параметры
    ys = []  # ответы

    # последняя итерация на элементе, индекс которого:
    end_index = len(dataset) - history_size - target_size - 1

    for i in range(0, end_index):
        # берем строки с индексом от `i` до `i + history_size`
        indexes = range(i, i + history_size)

        # добавляем их в массив с параметрами
        xs.append(dataset[indexes])

        # добавляем ответ из строки `текущая + target`
        ys.append(dataset[i + target_size][column_index])

    return np.array(xs), np.array(ys)
