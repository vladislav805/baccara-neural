import tensorflow as tf

from config import TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET, BUFFER_SIZE, BATCH_SIZE, PREDICT_COLUMN_INDEX
from data_utils import multivariate_data
from plot_utils import plot_train_history


# Создание и обучение модели с нуля
def create_and_train_model(dataset,
                           epochs: int,
                           steps: int,
                           validation_data,
                           single_step: bool = False,
                           checkpoint_dir: str = None,
                           ):
    # Подготавливаем данные для обучения
    x_train, y_train = multivariate_data(
        dataset,  # dataset
        dataset[:, PREDICT_COLUMN_INDEX],  # target
        0,  # start index
        TRAIN_SPLIT,  # end index
        PAST_HISTORY,  # history size
        FUTURE_TARGET,  # target size
        single_step=single_step)

    # Создание датасета для обучения
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data \
        .cache() \
        .shuffle(BUFFER_SIZE) \
        .batch(BATCH_SIZE) \
        .repeat()

    # Создание модели
    model = create_model(x_train, single_step)

    # Колбеки. Используется один для сохранения состояния модели при окончании эпохи
    callbacks = []

    # Если указана директория для сохранения модели, то добавляем колбек
    if checkpoint_dir is not None:
        # Создание колбека для сохранения чекпонитов при обучении
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir,
            save_weights_only=False,
            verbose=1
        ))

    # Обучение модели
    model_history = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=validation_data,
        callbacks=callbacks,
        validation_steps=50,
        use_multiprocessing=True,
    )

    # Вывод графика "как обучилоссс"
    plot_train_history(
        model_history,
        '{} step training and validation loss'.format('Single' if single_step else 'Multi')
    )
    
    return model


def create_model(x_train, single: bool):
    # Создание модели
    model = tf.keras.models.Sequential()

    if single:
        # Если точечное
        model.add(tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:]))
        model.add(tf.keras.layers.Dense(1))  # один выход - предсказание столбца

        # Компиляция модели
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    else:
        # Если интервальное
        model.add(tf.keras.layers.LSTM(32,
                                       return_sequences=True,
                                       input_shape=x_train.shape[-2:]))
        model.add(tf.keras.layers.LSTM(16, activation='relu'))
        model.add(tf.keras.layers.Dense(FUTURE_TARGET))  # НЕ ТОЧНО

        # Компиляция модели
        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    return model
