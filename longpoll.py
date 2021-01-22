import os
from typing import Tuple, Union, List, Optional

import tensorflow as tf
import pandas as pd
from time import time

# Названия колонок, которые будут использоваться как параметры
features_considered = [
    'cardPlayer1', 'cardPlayer2', 'mastPlayer1', 'mastPlayer2',
    'cardBanker1', 'cardBanker2', 'mastBanker1', 'mastBanker2',
]


def get_last_signal_id(dataset) -> int:
    return int(dataset[-1:]['id'])


# Получить датасет, по которому будем обучать
def get_train_data() -> Tuple[pd.Series, int]:
    zip_path = tf.keras.utils.get_file(
        # origin='http://longpoll.ru/dev/export.php?since=51587&count=54158',
        origin='/home/vlad805/www/__cat_long/result.csv',
        fname='data_export')

    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)

    last_signal_id = get_last_signal_id(df)

    # Получение только выбранных колонок (параметров)
    features = df[features_considered]
    features.index = df['id']  # добавление индекса

    return features.values, last_signal_id


# Получить последние сигналы
def get_last_signals() -> Tuple[Union[pd.Series, list], Optional[int]]:
    try:
        ncrnd = str(time())
        path = tf.keras.utils.get_file(
            'last_games' + ncrnd,
            origin='http://longpoll.ru/dev/export.php?' + ncrnd,
        )

        last_games = pd.read_csv(path)

        last_signal_id = get_last_signal_id(last_games)

        last_games = last_games[features_considered]
        last_games.index = last_games['id']

        return last_games.values, last_signal_id
    except Exception:
        return [], None


def get_signals_in_range(start, end) -> Tuple[Union[pd.Series, list], Optional[int]]:
    try:
        path = tf.keras.utils.get_file(
            'games_since_{0}_{1}'.format(start, end),
            origin='http://longpoll.ru/dev/export.php?since={0}&count={1}'.format(start, end - start),
        )

        data = pd.read_csv(path)

        last_signal_id = get_last_signal_id(data)

        games = data[features_considered]
        games.index = data['id']

        return games.values, last_signal_id
    except Exception:
        return [], None
