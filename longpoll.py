import tensorflow as tf
import pandas as pd
from time import time


# Получить последние сигналы
def get_last_signals():
    ncrnd = str(time())
    path = tf.keras.utils.get_file(
        'last_games' + ncrnd,
        origin='http://longpoll.ru/pythonSelectLastGames.php?' + ncrnd,
    )

    return pd.read_csv(path)
