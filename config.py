import matplotlib as mpl

# Глобальная настройка графиков
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# URL на получение датасета
DATASET_URL = 'http://longpoll.ru/pythonCSV.php?limit=51587&offset=54158'

# Сколько данных мы используем для обучения
TRAIN_SPLIT = 30000

# True - обучаем сеть, False - загружаем обученное
TRAIN = True

# Хотим интервальные чи нет?
INTERVAL_PREDICT = True

# Шагов в эпохе
STEPS_PER_EPOCH = 1440

# Количество эпох
EPOCHS = 20

BATCH_SIZE = 256
BUFFER_SIZE = 10000

# Где находится натренирванная сеть
CHECKPOINT_PATH = "training_test/cp.ckpt"

# Сколько прошлых строк используется для вычисления ответа (и обучения)
PAST_HISTORY = 720

# Сколько будущих строк будем пытаться предсказать?
FUTURE_TARGET = 15 if INTERVAL_PREDICT else 150

# С каким шагом берём выборку?
STEP = 1
