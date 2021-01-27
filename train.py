import matplotlib.pyplot as plt
import tensorflow as tf

from config import *
from data import multivariate_dataset, split_dataset
from longpoll import get_train_data
from model import get_model
from normalize import normalize_dataset

DATASET_TRAIN_SPLIT = 24000

# tf.keras.backend.set_floatx('float64')


def get_normalization_params():
    dataset, _ = get_train_data()
    train, _ = split_dataset(dataset, DATASET_TRAIN_SPLIT)
    _, mean, sd = normalize_dataset(train)
    return mean, sd


def do_train(_: None):
    # скачали и подготовили датасет
    dataset, _ = get_train_data()

    train, validate = split_dataset(dataset, DATASET_TRAIN_SPLIT)

    # разбили данные для обучения
    train_x, train_y = multivariate_dataset(
        dataset=train,
        column_index=DATASET_COLUMN_INDEX,
        history_size=HISTORY_SIZE,
        target_size=TARGET_SIZE,
    )

    # нормализовали данные для обучения
    train, mean, sd = normalize_dataset(train)

    # разбили данные для валидации
    validate_x, validate_y = multivariate_dataset(
        dataset=validate,
        column_index=DATASET_COLUMN_INDEX,
        history_size=HISTORY_SIZE,
        target_size=TARGET_SIZE,
    )

    validate, _, _ = normalize_dataset(validate, mean, sd)

    batch = 256

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
        .shuffle(1024)\
        .batch(batch) \
        .cache()\
        .repeat()

    validate_dataset = tf.data.Dataset.from_tensor_slices((validate_x, validate_y))\
        .batch(batch)\
        .repeat()

    print(validate_y)

    model = get_model(path=MODEL_TRAINED_PATH,
                      input_shape=(720, 8),
                      train=True)

    model_history = model.fit(
        train_dataset,
        validation_data=validate_dataset,
        validation_steps=50,
        epochs=50,
        steps_per_epoch=1440,
        # initial_epoch=10,
        # batch_size=1,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_TRAINED_PATH,
                save_weights_only=False,
                save_best_only=True,
            )
        ],
        use_multiprocessing=True,
    )

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(len(loss))
    plt.figure(1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.legend(loc='upper left')
    plt.show()



