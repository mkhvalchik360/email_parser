from functools import wraps
import logging
import os
from time import time
import configparser

timer_functions = {}

# Loading configuration from config file
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if f.__name__ in timer_functions.keys():
            current_elapsed_time = timer_functions[f.__name__]
        else:
            current_elapsed_time = 0
        timer_functions[f.__name__] = current_elapsed_time + (te - ts)
        logging.debug('func:%r took: %2.4f sec' % \
                        (f.__name__, te - ts))
        return result
    return wrap


def f_read_config(path=None):
    """ read config file from specified file path
    
    :param path: file path
    :return: configparser object
    """
    # Loading configuration from config file
    config = configparser.ConfigParser()
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(path, encoding='utf-8')
    return config

def f_setup_logger(level_sysout=logging.INFO, level_file=logging.DEBUG, folder_path="logs"):
    """Setup logger

    By default we display only INFO in console, and write everything in file

    Args:
        level_sysout: Level that is displayed in console (default INFO)
        level_file: Level that is written in file (default DEBUG)

    Returns:
        Nothing

    """
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(filename=os.path.join(folder_path, "amf_uce_nlp_{}.log".format(time())),
                                       encoding='utf-8')
    sysout_handler = logging.StreamHandler()
    file_handler.setLevel(level_file)
    sysout_handler.setLevel(level_sysout)
    logging.basicConfig(handlers=[file_handler, sysout_handler], level=logging.DEBUG,
                        format='%(asctime)s (%(levelname)s) %(message)s', datefmt='%m/%d/%y %I:%M:%S %p')


def get_model_full_path(model_name):
    path_models = config["DEFAULT"]["path_models"]
    return os.path.join(os.path.dirname(__file__), path_models, model_name)