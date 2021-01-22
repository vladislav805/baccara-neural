import sys

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow
tensorflow.random.set_seed(13)

from train import do_train
from predict import do_predict, do_mass_test
from bot import do_bot

commands = {
    'train': do_train,
    'predict': do_predict,
    'bot': do_bot,
    'mass_test': do_mass_test,
}

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) == 0:
        sys.stderr.write('no pass command')
        exit(1)

    command: str = args.pop(0)

    if command in commands:
        commands[command](args)
    else:
        sys.stderr.write('command not found')
        exit(1)
