import tensorflow as tf
import pandas as pd


# Получить последние сигналы
def get_last_signals():
    path = tf.keras.utils.get_file(
        'last_games',
        origin='http://longpoll.ru/pythonSelectLastGames.php',
    )

    return pd.read_csv(path)
