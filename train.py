import tensorflow as tf
import os
import pandas as pd
from plot_utils import *
from data import *
from train_utils import create_and_train_model
from config import DATASET_URL, TRAIN_SPLIT, CHECKPOINT_PATH, EPOCHS, \
    STEPS_PER_EPOCH, PAST_HISTORY, STEP, FUTURE_TARGET, BATCH_SIZE, TRAIN, INTERVAL_PREDICT

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
    'cardBanker1', 'cardBanker2', 'mastBanker1', 'mastBanker2'
]

# Получение только выбранных колонок (параметров)
features = df[features_considered]
features.index = df['id']  # добавление индекса

dataset = features.values

# Среднее значение
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)

# Стандартное отклонение
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

# Нормализация данных
dataset = (dataset - data_mean) / data_std

# Путь к директории с чекпоинтами
CHECKPOINT_DIRECTORY = os.path.dirname(CHECKPOINT_PATH)

x_validation, y_validation = multivariate_data(
    dataset,  # dataset
    dataset[:, 1],  # target
    TRAIN_SPLIT,  # start index
    None,  # end index
    PAST_HISTORY,  # history size
    FUTURE_TARGET,  # target size
    STEP,  # step
    single_step=True)

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
                                   interval_predict=INTERVAL_PREDICT,
                                   checkpoint_dir=CHECKPOINT_DIRECTORY)
else:
    # Если не обучаем, значит тупо загружаем
    model = tf.keras.models.load_model(CHECKPOINT_PATH)

if INTERVAL_PREDICT:
    for x, y in validation_data.take(3):
        """
        print('##########################################')
        print('y=', y[0].numpy())
        
        print('predict_result=', predict_result)
        norm_value = predict_result * data_std[1] + data_mean[1]
        print('norm_value=', norm_value)
        """

        predict_result = model.predict(x)[0]

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
        multi_step_plot(x[0], y[0], np.array([0]))
