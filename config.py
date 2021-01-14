import matplotlib as mpl

# Глобальная настройка графиков
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# URL на получение датасета
DATASET_URL = 'http://longpoll.ru/pythonCSV.php?limit=51587&offset=54158'

#########
# Общее #
#########
# True - обучаем сеть, False - загружаем обученное
TRAIN = False

# Хотим точечный прогноз (True) или интервальный (False)?
SINGLE_STEP = True

# Индекс колонки, которую мы хотим предсказать
PREDICT_COLUMN_INDEX = 1

############
# Обучение #
############
# Шагов в эпохе
STEPS_PER_EPOCH = 1440

# Сколько данных мы используем для обучения
TRAIN_SPLIT = 40000

# Количество эпох
EPOCHS = 20

BATCH_SIZE = 256
BUFFER_SIZE = 10000

# Где находится натренирванная сеть
CHECKPOINT_PATH = "training/cp.ckpt"

# Сколько прошлых строк используется для вычисления ответа (и обучения)
PAST_HISTORY = 720

# Сколько будущих строк будем пытаться предсказать?
# 1 - если мы хотим точечно предсказывать
# 60 (или любое другое число) - если мы хотим спрогнозировать сразу этого период
FUTURE_TARGET = 1 if SINGLE_STEP else 60
