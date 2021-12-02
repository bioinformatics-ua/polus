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

__version__="0.1.6"

# add main lib sub packages
import callbacks
import core
import data
import layers
import losses
import metrics
import models
import schedulers
import training
import utils
import ir
import ner
import experimental