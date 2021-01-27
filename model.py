import os.path
from typing import Optional, Tuple

import tensorflow as tf



def get_model(path: str = None,  # Путь к модели
              input_shape: Optional[Tuple[int, int]] = None,  # Размерность входа
              train: bool = False,  # Нужно ли её обучать или просто прочитать?
              ) -> tf.keras.models.Model:
    exists = os.path.exists(path) if path is not None else False
    if not train:
        if exists:
            return tf.keras.models.load_model(path)
        else:
            raise FileNotFoundError('Trained model on path {} not found'.format(path))
    else:
        if exists:
            answer = input('Model on path {} already exists. Rewrite? (y - continue): ')
            if answer != 'y' and answer != 'н':
                exit(0)

        if path is not None:
            return create_model(input_shape=input_shape)
        else:
            raise FileNotFoundError('path not specified')


# Просто создаём модель
def create_model(input_shape):
    # Создание модели
    model = tf.keras.models.Sequential()
    # , activation='softmax'
    model.add(tf.keras.layers.LSTM(16, input_shape=input_shape, dropout=0.1, activation='softmax'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(13))
    # model.add(tf.keras.layers.BatchNormalization())

    # Компиляция модели
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005), loss='categorical_crossentropy')

    return model
