import tensorflow as tf
import tensorflow_addons as tfa

import os
import json
import h5py
import pickle
import types

from polus.core import BaseLogger
from functools import wraps
from polus.utils import merge_dicts, flatten_dict

#import for refering to this file, used in the load_model method
import polus.models


from transformers import TFBertModel

def load_model(file_name_w_ext, change_config={}, external_module=None):
    
    file_name = os.path.splitext(file_name_w_ext)[0]
    
    with open(file_name_w_ext,"r") as f:
        cfg = json.load(f)
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    if external_module is not None:
        model = getattr(external_module, cfg['func_name'])(**cfg)
    else:
        model = getattr(polus.models, cfg['func_name'])(**cfg)
    
    # correctly init the model from samples if given
    if os.path.exists(file_name+".init"):
        with open(file_name+".init", "rb") as f:
            args, kwargs = pickle.load(f)
        self.logger.info("Init the loaded model")
        model.init_from_data(*args, **kwargs)
    
    # load weights
    with h5py.File(file_name+".h5", 'r') as f:
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
        
        model = func(**flatten_dict(kwargs))
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
        
        if hasattr(self, "_init"):
            with open(path+".init","wb") as f:
                pickle.dump(self._init, f)
        
        #save model weights
        with h5py.File(path+".h5", 'w') as f:
            weight = self.get_weights()
            for i in range(len(weight)):
                f.create_dataset('weight'+str(i), data=weight[i])
    
    def init_from_data(self, *args, **kwargs):
        """
        The model was not yet built and therefore, it needs to
        receive some correct samples in order to lazy init all
        of the layers
        """
        
        ## init from training data
        
        # store the samples that are given to the model
        self._init = (args, kwargs)
        return self(*args, **kwargs)
    
    def set_name(self, name):
        self._name = name
        
class SequentialSavableModel(tf.keras.Sequential, SavableModel):
    """
    This class is just to be compatible with the Sequential Model from keras and implements
    the same inference method from the SavableModel class
    """
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)
