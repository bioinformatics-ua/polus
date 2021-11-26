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
    
    def set_logging_level(self, logging_level):
        self.logger.setLevel(logging_level)
        
        
def find_dtype_and_shapes(data_generator, k=5):
    """
    Automatically gets the dtype and shapes of samples
    
    """
    # get one sample
    assert k>0
    generator = iter(data_generator)
    samples = [ next(generator) for i in range(k) ]

    
            
    if isinstance(samples[0], dict):
        
        dtypes = {}
        shapes = {}

        for key in samples[0].keys():
            tf_value = tf.constant(samples[0][key])
            dtypes[key] = tf_value.dtype
            shapes[key] = tf_value.shape


        # infer the shape since it can be None if some dim is not in agreement 
        for i in range(len(samples)-1):
            assert len(set(samples[i].keys()) - set(samples[i+1].keys())) == 0
            for key in samples[i+1].keys():
                tf_value = tf.constant(samples[i+1][key])

                # they must have the same dimensionality, but can have diff values per dimension
                assert len(tf_value.shape) == len(shapes[key])

                # sample with diff value in one of the dimensions
                if tf_value.shape != shapes[key]:
                    new_shape = list(shapes[key])
                    for j in range(len(shapes[key])):
                        if shapes[key][j] is not None and shapes[key][j]!=tf_value.shape[j]:
                            new_shape[j] = None
                    shapes[key] = tf.TensorShape(new_shape)

    else:
        raise ValueError(f"The find_dtype_and_shapes only supports when the sample came from generator are dict but found {type(samples[0])}")
    
    return dtypes, shapes


def execute_if(condition_var, error_message="", on=True):
    def decorator(func):
        def function_wrapper(self, *args, **kwargs):

            if getattr(self, condition_var) == on:
                return func(self, *args, **kwargs)
            else:
                if error_message != "":
                    print(error_message)
        return function_wrapper
    
    return decorator
