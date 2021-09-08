import logging
import os
import sys
import tensorflow as tf
from logging.handlers import TimedRotatingFileHandler

"""
Describe code that is used through all packages of this toolkit
"""

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

# module-wide variables
this = sys.modules[__name__]
this.jit_compile=False

def activate_jit_compile():
    this.jit_compile=True
    
def set_jit_compile(mode):
    this.jit_compile=mode

def get_jit_compile():
    return this.jit_compile
    
class BaseLogger:
    def __init__(self, logging_level=logging.DEBUG, log_name="polus.log"):
        """
        From: https://www.toptal.com/python/in-depth-python-logging
        """
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging_level)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(FORMATTER)
            self.logger.addHandler(console_handler)

            if not os.path.exists('logs'):
                os.makedirs('logs')

            file_handler = TimedRotatingFileHandler(os.path.join("logs", log_name), when='midnight', encoding='utf-8')
            file_handler.setFormatter(FORMATTER)
            self.logger.addHandler(file_handler)

            self.logger.propagate = False
    
    def set_logging_level(logging_level):
        self.logger.setLevel(logging_level)
        
        
def find_dtype_and_shapes(data_generator):
    """
    Automatically gets the dtype and shapes of samples
    
    """
    # get one sample
    sample = next(iter(data_generator))

    if isinstance(sample, dict):
        dtypes = {}
        shapes = {}
        for key in sample.keys():
            tf_value = tf.constant(sample[key])
            dtypes[key] = tf_value.dtype
            shapes[key] = tf_value.shape
    elif isinstance(sample, tuple):
        dtypes = []
        shapes = []
        for e in sample:
            tf_value = tf.constant(e)
            dtypes.append(tf_value.dtype)
            shapes.append(tf_value.shape)
        dtypes = tuple(dtypes)
        shapes = tuple(shapes)
    elif isinstance(sample, list):
        dtypes = []
        shapes = []
        for e in sample:
            tf_value = tf.constant(e)
            dtypes.append(tf_value.dtype)
            shapes.append(tf_value.shape)

    else:
        raise ValueError(f"The find_dtype_and_shapes only supports when the sample from generator are dict or tuples or list, but found {type(sample)}")
    
    return dtypes, shapes


