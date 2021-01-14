import os
import numpy as np
import pandas as pd
import tensorflow as tf

from config import DATASET_URL, TRAIN_SPLIT, CHECKPOINT_PATH, EPOCHS, \
    STEPS_PER_EPOCH, PAST_HISTORY, FUTURE_TARGET, BATCH_SIZE, \
    TRAIN, SINGLE_STEP, PREDICT_COLUMN_INDEX
from data_utils import multivariate_data, normalize_dataset, normalize_value
from plot_utils import show_plot, multi_step_plot
from train_utils import create_and_train_model

# Загрузка датасета
zip_path = tf.keras.utils.get_file(origin=DATASET_URL, fname='data_export')
csv_path, _ = os.path.splitext(zip_path)

# Чтение датасета
df = pd.read_csv(csv_path)

# Инициализация рандома
tf.random.set_seed(13)

# Названия колонок, которые будут использоваться как параметры
features_considered = [
    'cardPlayer1', 'cardPlayer2', 'mastPlayer1', 'mastPlayer2',
    'cardBanker1', 'cardBanker2', 'mastBanker1', 'mastBanker2',
]

# Получение только выбранных колонок (параметров)
features = df[features_considered]
features.index = df['id']  # добавление индекса

# нормализация датасета, среднее значение и стандартное отклонение (последние
# два нужны для нормализации результата)
dataset, mean, sd = normalize_dataset(features.values, TRAIN_SPLIT)

# Подготовка данных для валидации
x_validation, y_validation = multivariate_data(
    dataset,  # dataset
    dataset[:, PREDICT_COLUMN_INDEX],  # target
    TRAIN_SPLIT,  # start index
    None,  # end index
    PAST_HISTORY,  # history size
    FUTURE_TARGET,  # target size
    single_step=SINGLE_STEP)

# Данные для сверки/проверки/валидации
validation_data = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))\
    .batch(BATCH_SIZE)\
    .repeat()

# Если обучаем
if TRAIN:
    model = create_and_train_model(dataset,
                                   epochs=EPOCHS,
                                   steps=STEPS_PER_EPOCH,
                                   validation_data=validation_data,
                                   single_step=SINGLE_STEP,
                                   checkpoint_dir=os.path.dirname(CHECKPOINT_PATH))
else:
    # Если не обучаем, значит тупо загружаем
    model = tf.keras.models.load_model(CHECKPOINT_PATH)

if SINGLE_STEP:
    for x, y in validation_data.take(3):
        predict_result = model.predict(x)[0]

        normalized_value = normalize_value(predict_result, mean, sd, PREDICT_COLUMN_INDEX)
        print('norm_value=', normalized_value)

        plot = show_plot(
            [
                x[0][:, 1].numpy(),
                y[0].numpy(),
                predict_result,
            ],
            12,
            'Single Step Prediction'
        )
        plot.show()
else:
    for x, y in validation_data.take(1):
        pred = model.predict(x)[0]
        print('x', x[0].shape)
        print('y', y[0].shape)
        print('pred', pred.shape)
        multi_step_plot(
            np.array(x[0]),
            np.array(y[0]),
            pred)
