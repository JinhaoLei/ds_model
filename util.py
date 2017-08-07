import os

import config


def load_word():
    words = [line.split()[0] for line in open(config.vocab_file)]
    return words


def load_word_to_id():
    words = load_word()
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_checker(file_name):
    def decorator(func):
        def func_with_file_check(*args, **kargs):
            print ('preparing')
            print(file_name)
            if not os.path.exists(file_name):
                func(*args, **kargs)
            print (file_name)
            print('prepared')
        return func_with_file_check
    return decorator
