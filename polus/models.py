import tensorflow as tf
import tensorflow_addons as tfa

import os
import json
import h5py

import types

from polus.core import BaseLogger
from functools import wraps
from polus.utils import merge_dicts

#import for refering to this file, used in the load_model method
import polus.models


from transformers import TFBertModel

def load_model(file_name, change_config={}, external_module=None):
    
    with open(file_name,"r") as f:
        cfg = json.load(f)
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    if external_module is not None:
        model = getattr(external_module, cfg['func_name'])(**cfg)
    else:
        model = getattr(polus.models, cfg['func_name'])(**cfg)
        
    # load weights
    with h5py.File(file_name.split(".")[0]+".h5", 'r') as f:
        weight = []
        for i in range(len(f.keys())):
            weight.append(f['weight'+str(i)][:])
        model.set_weights(weight)
    
    return model

        
def resolve_activation(activation_name):
    if activation_name=="mish":
        return tfa.activations.mish
    else:
        return activation_name
    

def from_config(func):
    @wraps(func)
    def function_wrapper(**kwargs):
        
        # run resolve_activation
        if "activation" in kwargs["model"]:
            # cache the reference for the activation
            _activation = kwargs["model"]["activation"]
            kwargs["model"]["activation"] = resolve_activation(_activation)
        
        model = func(**kwargs["model"])
        kwargs['func_name'] = func.__name__
        
        # restore the original activation reference for saving
        if "activation" in kwargs["model"]:
            kwargs["model"]["activation"] = _activation
        
        model._name = func.__name__
        model.savable_config = kwargs

        return model

    return function_wrapper


class SavableModel(tf.keras.Model, BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # This class also extends BaseLogger, but Keras last subclass do not call super 
        # so it must be manually called
        BaseLogger.__init__(self)
    
    def save(self, base_path = os.path.join(".polus_cache","saved_models"), extension=""):
        
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        path = os.path.join(base_path, self.name+extension)

        cfg = self.savable_config
        with open(path+".cfg","w") as f:
            json.dump(self.savable_config , f)

        #save model weights
        with h5py.File(path+".h5", 'w') as f:
            weight = self.get_weights()
            for i in range(len(weight)):
                f.create_dataset('weight'+str(i), data=weight[i])
    
    def set_name(self, name):
        self._name = name
        
class SequentialSavableModel(tf.keras.Sequential, SavableModel):
    """
    This class is just to be compatible with the Sequential Model from keras and implements
    the same inference method from the SavableModel class
    """
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)
