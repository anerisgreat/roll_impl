import os
from logging import config as loggingconfig
import logging
import datetime
from functools import partial

def joinmakedir(a, b):
    newp = os.path.join(a, b)
    os.makedirs(newp, exist_ok = True)
    return newp

def logging_get_default_config(
        debug_fname = None, console_level = logging.INFO):
    base_dict  = \
    {
        "version":1,
        "root":{
            "handlers" : ["console"],
            "level" : logging.DEBUG
        },
        "handlers":{
            "console":{
                "formatter": "stdout",
                "class": "logging.StreamHandler",
                "level": console_level
            },
        },
        "formatters":{
            "stdout": {
                "format": "%(asctime)s : %(levelname)8s : "\
                    "%(module)10s - %(message)s",
                "datefmt":"%d-%m-%Y %I:%M:%S"
            },
            "debug" : {
                "format": "%(asctime)s : %(levelname)8s : %(module)10s : "\
                    "%(funcName)10s : %(lineno)4d : (Process Details : "\
                    "(%(process)d, %(processName)s), Thread Details : "\
                    "(%(thread)d, %(threadName)s))\nLog : %(message)s",
                "datefmt":"%d-%m-%Y %I:%M:%S"
            }
        },
    }

    if not debug_fname is None:
        base_dict['root']['handlers'] += ['debug']
        base_dict['handlers']['debug'] = \
            {\
             "class" : "logging.FileHandler",
             "formatter" : "debug",
             "filename" : debug_fname,
             "level" : logging.DEBUG}
        return base_dict

def init_experiment(base_dir, experiment_name):
    experiment_dir = joinmakedir(base_dir, experiment_name)
    run_dir = joinmakedir(
        experiment_dir,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))

    loggingconfig.dictConfig(logging_get_default_config(
        debug_fname = os.path.join(run_dir, 'debug.log'),
        console_level = logging.INFO))
    #INIT LOG
    return run_dir

