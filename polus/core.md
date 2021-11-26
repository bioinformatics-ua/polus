Module polus.core
=================
# Polus core 

Here we describe operations that are used through all packages 
of this toolkit.

Furthermore, the package core has module-wide variables that 
govern some internal behaviour of this framework. For instance, 
polus.core has a jit_compile attribute that controls 
the utilization of XLA during the code conversion to static 
computational graphs, the correct way to write or read this 
attribute is through its associated methods:

```python

from polus.core import set_jit_compile, get_jit_compile

print(get_jit_compile()) # True
set_jit_compile(False) #False
print(get_jit_compile()) # False

```

Functions
---------

    
`execute_if(condition_var, error_message='', on=True)`
:   A decorated function that restricts the execution of a specific
    function based on a condition. This is useful for instance methods
    that we only want to be called based on a simple condition.
    
    Args:
      condition_var (boolean): Boolean expression that will be evaluated
        when the decorated method is called.
      
      error_message (str): A printable error message to be displayed when
        the condition fails.
        
      on (boolean): The condition that the *condition_var* is evaluated
        against, by default the decorated method executes if *condition_var*
        is True.

    
`find_dtype_and_shapes(data_generator, k=5)`
:   Automatically infer the data type and shapes of the samples from 
    the *data_generator*. This function is highly used in polus.data.
    
    Args:
      data_generator (python generator): python generator that output
        samples in a **dictionary** format.
        
      k (int): Number of samples that are output from *data_generator*
        in order to correctly infer the shape of each dictionary value.
        For instance, if a sample contains a dynamic shape, this function
        will return None in the place of the dimension that as a dynamic 
        shape.

    
`get_jit_compile()`
:   Reads the value of the *jit_compile* attribute.

    
`set_jit_compile(mode: bool)`
:   Changes the *jit_compile* attribute.
    
    Args:
      mode (bool): Boolean variable that defines the utilization of
        jit_compiler during build of computation graphs, which enables
        XLA compiler.
        
    Returns:
      None

Classes
-------

`BaseLogger(logging_level=10, log_name='polus.log')`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Descendants

    * polus.callbacks.ICallback
    * polus.callbacks.IOutput
    * polus.data.DataLoader
    * polus.data.IAccelerated_Map
    * polus.layers.CRF
    * polus.metrics.IMetric
    * polus.models.SavableModel
    * polus.ner.utils.BioCSequenceDecoder
    * polus.training.BaseTrainer

    ### Methods

    `set_logging_level(self, logging_level)`
    :   Args:
          logging_level (int): A valid logging level supperted by the
            logging package, see https://docs.python.org/3/library/logging.html#logging-levels
            
        Returns:
          None