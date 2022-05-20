r'''

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

'''

import logging
import os
import sys
import tensorflow as tf



def set_jit_compile(mode: bool):
    """
    Changes the *jit_compile* attribute.
    
    Args:
      mode (bool): Boolean variable that defines the utilization of
        jit_compiler during build of computation graphs, which enables
        XLA compiler.
        
    Returns:
      None
    """
    os.environ["POLUS_JIT"]=str(mode)

def get_jit_compile():
    """
    Reads the value of the *jit_compile* attribute.
    """
    
    if os.environ.get("POLUS_JIT") is None:
        set_jit_compile(False) # default mode
    
    return os.environ.get("POLUS_JIT")=="True"
        
def find_dtype_and_shapes(data_generator, k=10):
    """
    Automatically infer the data type and shapes of the samples from 
    the *data_generator*. This function is highly used in polus.data.
    
    Args:
      data_generator (python generator): python generator that output
        samples in a **dictionary** format.
        
      k (int): Number of samples that are output from *data_generator*
        in order to correctly infer the shape of each dictionary value.
        For instance, if a sample contains a dynamic shape, this function
        will return None in the place of the dimension that as a dynamic 
        shape. If set to -1 it will read the entire generator
    """
    
    
    
    if k==-1:
        samples = [ sample for sample in data_generator ]
    elif k>0:
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
    """
    A decorated function that restricts the execution of a specific
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
    
    """
    def decorator(func):
        def function_wrapper(self, *args, **kwargs):

            if getattr(self, condition_var) == on:
                return func(self, *args, **kwargs)
            else:
                if error_message != "":
                    print(error_message)
        return function_wrapper
    
    return decorator


    