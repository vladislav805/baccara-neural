import tensorflow as tf
import numpy as np
from typing import List

from model import get_model
from config import *
from longpoll import get_last_signals, get_signals_in_range
from train import get_normalization_params
from normalize import normalize_value, normalize_dataset
from data import split_dataset


# С какого сигнала брать (использовать только для тестов, если последних сигналов недостаточно, иначе None)
from utils import vector_to_card

SINCE = 147141


def predict_by_last_signals(model,
                            dataset,
                            mean: List[float],
                            sd: List[float]):
    last_signals = dataset
    #last_signals, _, _ = normalize_dataset(dataset, mean, sd)

    # Формируем датасет
    dataset = tf.data.Dataset.from_tensors(last_signals) \
        .batch(256) \
        .repeat()

    # Берем из датасета одну матрицу
    item = dataset.take(1)

    if model is None:
        model = get_model(MODEL_TRAINED_PATH)

    predict = vector_to_card(model.predict(item)[0])

    # Предсказываем
    predicted = normalize_value(
        predict,
        mean[DATASET_COLUMN_INDEX],
        sd[DATASET_COLUMN_INDEX])

    return predicted, int(np.fix(predicted))


def do_predict(_):
    real = None

    # Если не укзано с какого сигнала брать (для тестов), то берем последние
    # Иначе берем от SINCE до SINCE + 720
    if SINCE is None:
        last_signals, last_signal_id = get_last_signals()
    else:
        last_signals, last_signal_id = get_signals_in_range(SINCE, SINCE + HISTORY_SIZE + 1)
        last_signals, real = split_dataset(last_signals, by=HISTORY_SIZE)

    # Получаем номер следущего сигнала
    target_signal_id = last_signal_id + 1

    # Вычисляем коэффицинты нормализации из обучающего датасета
    mean, sd = get_normalization_params()

    # Создаём и загружаем модель
    model = get_model(path=MODEL_TRAINED_PATH)

    # Предсказываем
    predicted = predict_by_last_signals(model, last_signals, mean, sd)

    if SINCE is not None:
        print('predicted =', predicted, 'real =', real[0][DATASET_COLUMN_INDEX])
    else:
        print('in id = ', target_signal_id, 'predicted = ', predicted)


MASS_TEST_SINCE = 147141
MASS_TEST_ITERATIONS = 250


def do_mass_test(_):
    mean, sd = get_normalization_params()
    passed = 0

    # Создаём и загружаем модель
    model = get_model(path=MODEL_TRAINED_PATH)

    for row_id in range(MASS_TEST_SINCE, MASS_TEST_SINCE + MASS_TEST_ITERATIONS):
        last_signals, last_signal_id = get_signals_in_range(row_id, row_id + HISTORY_SIZE + 1)
        last_signals, real = split_dataset(last_signals, by=HISTORY_SIZE)

        predicted, predicted_value = predict_by_last_signals(model=model,
                                                             dataset=last_signals,
                                                             mean=mean,
                                                             sd=sd)
        val = real[0][DATASET_COLUMN_INDEX]
        print('predicted =', predicted, 'value =', predicted_value, 'real =', val)
        if val == predicted_value:
            passed = passed + 1

    print('passed ', passed, 'of', MASS_TEST_ITERATIONS, ', ', round(passed * 100 / MASS_TEST_ITERATIONS, 2), '%')
