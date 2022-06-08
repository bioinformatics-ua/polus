r'''
# Welcome to Polus!!!

Polus is a powerful tensorflow toolkit for creating/training 
complex deep learning models in a functional way.

This toolkit is currently under development and aims to focus on academic research,
like biomedical tasks, although it can also also be used in other domains.

# Main packages

Polus consists of a main API that resides under the polus package
and more tasks specific APIs that will reside under the task name.
For instance, polus.ner and polus.ir are sub-packages that are
focused on ner (named entity recognition) and ir (information 
retrieval) tasks. 

## The main API

The main API consists of:

- `polus.training.py`: Training API that contains the most basic training loop
- `polus.data.py`: DataLoaders API that extends the tf.data.Dataset functionality
 to build more useful data loaders with easy to use caching mechanisms.
- `polus.callbacks.py`: Main source of interaction with the main training loop
- `polus.models.py`: Model API define an extension of the tf.keras.Model by handling
 storing and loading of the entire model architecture.
- `polus.metrics`: Metrics API describe how metrics should be implemented so that can
 be efficiently used during the training loop.
- `polus.core`: The Core API defines some base classes or variables that are used
 through the framework. It also exposes some functions to change the internal
 behaviour of the polus framework, e.g., the use of XLA.
 
## Remaining of the polus package

The remaining of the files not yet mentioned act as code repository and hold
some utility classes, e.g. `polus.layers.py` contains some ready to use layers that
can be imported and used by tf.keras.Model(s).

# Notebooks and examples

At the time of writing, there are no notebooks available... work in progress

# TensorFlow focused

Since this framework was designed from scratch with TensorFlow 2.3+ in mind
we leveraged the most recent features to make sure that the code runs
smoothly and fast as possible. For instance, internally we utilize static
computational graphs during the training procedure, and XLA is 
enabled by default, which can be easily accessed by the polus.core API.

'''

__version__="0.2.1"
import logging
from logging.handlers import TimedRotatingFileHandler

import os
import sys
# setting up logger
logger = logging.getLogger(__name__)

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s: %(message)s")
DEBUG_FORMATTER = logging.Formatter("%(asctime)s — %(filename)s:%(name)s:%(funcName)s:%(lineno)d: %(message)s")

if "POLUS_LOGGER_LEVEL" in os.environ:
    m = {"DEBUG": logging.DEBUG, 
         "INFO": logging.INFO, 
         "WARN": logging.WARN, 
         "ERROR": logging.ERROR}
    logger.setLevel(os.environ["POLUS_LOGGER_LEVEL"])
else:
    logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)

logger.addHandler(console_handler)

if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = TimedRotatingFileHandler(os.path.join("logs", "polus.log"), when='midnight', encoding='utf-8')
file_handler.setLevel(logging.WARN)
file_handler.setFormatter(FORMATTER)
logger.addHandler(file_handler)

file_handler_db = TimedRotatingFileHandler(os.path.join("logs", "debug.log"), when='midnight', encoding='utf-8')
file_handler_db.setLevel(logging.DEBUG)
file_handler_db.setFormatter(DEBUG_FORMATTER)
logger.addHandler(file_handler_db)

import tensorflow as tf

try:
    import horovod.tensorflow as hvd
except ModuleNotFoundError:
    import polus.mock.horovod as hvd

from polus.utils import Singleton
# init some vars
class PolusContext(metaclass=Singleton):
    
    def __init__(self):
        logger.debug("-----------------DEBUG INIT POLUS CONTEXT-------------")
        self.use_horovod = False
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 1:
            if hvd.init() == "mock":
                logger.info(f"The script found multiple GPUs, however it cannot use them since multi-gpu"
                                 f" requires horovod.tensorflow module to be installed.\n"
                                 f"Intead the process will only use one")
            else:
                if hvd.size() <= 1:
                    logger.info(f"The script found multiple GPUs and a horovod.tensorlfow installation. However,"
                                 f" only one process was initialized, please check if you are runing the script with horovodrun or mpirun.")
                else:
                    if hvd.local_rank() == 0:
                        logger.info(f"MultiGPU training enabled, using {hvd.size()} processes ")
                    self.use_horovod = True

            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            
    def is_horovod_enabled(self):
        return self.use_horovod
        
PolusContext()



# add main lib sub packages
#import polus.callbacks
#import polus.core
#import polus.data
#import polus.layers
#import polus.losses
#import polus.metrics
#import polus.models
#import polus.schedulers
#import polus.training
#import polus.utils#
#import polus.hpo
#import polus.ir
#import polus.ner
#import polus.experimental