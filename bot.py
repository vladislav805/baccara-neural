import requests
import os

from predict import predict_by_last_signals
from train import get_normalization_params
from longpoll import get_last_signals
from utils import set_interval, get_card_by_id
from config import *
from model import get_model


def do_bot(_):
    # Вычисляем коэффицинты нормализации из обучающего датасета
    mean, sd = get_normalization_params()

    # Создаём обработчик для бота
    bot_handler = bot_factory(mean, sd)

    bot_handler()
    set_interval(bot_handler, 30)


def bot_factory(mean, sd):
    # Создаём и загружаем модель
    model = get_model(path=MODEL_TRAINED_PATH)

    def handler():
        # Получаем предыдущие игры
        last_signals, last_signal_id = get_last_signals()

        if len(last_signals) == 0 and last_signal_id is None:
            print('not enough data')
            return

        # Предсказываем
        (predict_orig, predict_norm) = predict_by_last_signals(model=model,
                                                               dataset=last_signals,
                                                               mean=mean,
                                                               sd=sd)

        card = get_card_by_id(predict_norm)

        text = "Signal = {0}\nPredict = {1}\nNormalized predict = {2}\nCard = {3}".format(
            last_signal_id + 1,
            predict_orig,
            predict_norm,
            card,
        )

        for chat_id in BOT_TARGET_IDS:
            send_telegram_message(chat_id=chat_id, text=text)

    return handler


def send_telegram_message(chat_id: int, text: str):
    token = os.getenv('TG_TOKEN')

    if token is None:
        print('TG_TOKEN not set')
        exit(1)

    requests.get('https://api.telegram.org/bot' + token + '/sendMessage',
                 params={
                     'chat_id': chat_id,
                     'text': text
                 }).json()
