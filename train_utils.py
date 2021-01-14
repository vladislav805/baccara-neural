import tensorflow as tf
from plot_utils import plot_train_history
from data import multivariate_data
from config import TRAIN_SPLIT, PAST_HISTORY, FUTURE_TARGET, STEP, BUFFER_SIZE, BATCH_SIZE


def create_and_train_model(dataset,
                           epochs: int,
                           steps: int,
                           validation_data,
                           interval_predict: bool = False,
                           checkpoint_dir: str = None,
                           ):
    x_train, y_train = multivariate_data(
        dataset,  # dataset
        dataset[:, 1],  # target
        0, TRAIN_SPLIT,  # start / end indexes
        PAST_HISTORY,  # history size
        FUTURE_TARGET,  # target size
        STEP,  # step
        single_step=True)

    # Данные для обучения
    # Создание датасета для обучения
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data \
        .cache() \
        .shuffle(BUFFER_SIZE) \
        .batch(BATCH_SIZE) \
        .repeat()

    model = create_model(x_train, interval_predict=interval_predict)

    callbacks = []
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
        use_multiprocessing=True
    )

    plot_train_history(model_history, 'Single Step Training and validation loss')


def create_model(x_train,
                 interval_predict: bool
                 ):
    # Создание модели
    model = tf.keras.models.Sequential()

    if not interval_predict:
        # Если точечное
        model.add(tf.keras.layers.LSTM(32, input_shape=x_train.shape[-2:]))
        model.add(tf.keras.layers.Dense(1))

        # Компиляция модели
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    else:
        # Если интервальное
        model.add(tf.keras.layers.LSTM(32,
                                       return_sequences=True,
                                       input_shape=x_train.shape[-2:]))
        model.add(tf.keras.layers.LSTM(16, activation='relu'))
        model.add(tf.keras.layers.Dense(72))

        # Компиляция модели
        model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    return model
