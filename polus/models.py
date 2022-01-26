import tensorflow as tf
import tensorflow_addons as tfa

import os
import json
import h5py
import pickle
import types

from polus.core import BaseLogger,get_jit_compile
from functools import wraps
from polus.utils import merge_dicts, flatten_dict, complex_json_serializer, complex_json_deserializer

#import for refering to this file, used in the load_model method
import polus.models

def load_model(file_name_w_ext, change_config={}, external_module=None):
    
    file_name = os.path.splitext(file_name_w_ext)[0]
    
    with open(file_name_w_ext,"r") as f:
        cfg = complex_json_deserializer(json.load(f))
    
    cfg["model"] = merge_dicts(cfg["model"], change_config)
    
    if external_module is not None:
        model = getattr(external_module, cfg['func_name'])(**cfg)
    else:
        model = getattr(polus.models, cfg['func_name'])(**cfg)
    
    # correctly init the model from samples if given
    if os.path.exists(file_name+".init"):
        with open(file_name+".init", "rb") as f:
            args, kwargs = pickle.load(f)
        print("Init the loaded model")
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


class PolusModel(tf.keras.Model, BaseLogger):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # This class also extends BaseLogger, but Keras last subclass do not call super 
        # so it must be manually called
        BaseLogger.__init__(self)
    
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
        
class SavableModel(PolusModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def save(self, 
             base_path = os.path.join(".polus_cache","saved_models"), 
             extension=""):
        
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        path = os.path.join(base_path, self.name+extension)
        
        with open(path+".cfg","w") as f:
            json_str = complex_json_serializer(self.savable_config)
            json.dump(json_str , f)
        
        if hasattr(self, "_init"):
            with open(path+".init","wb") as f:
                pickle.dump(self._init, f)
        
        #save model weights
        with h5py.File(path+".h5", 'w') as f:
            weight = self.get_weights()
            for i in range(len(weight)):
                f.create_dataset('weight'+str(i), data=weight[i])
        
class SequentialSavableModel(tf.keras.Sequential, SavableModel):
    """
    This class is just to be compatible with the Sequential Model from keras and implements
    the same inference method from the SavableModel class
    """
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)


from transformers.modeling_tf_utils import shape_list
from transformers import TFBertModel, AutoTokenizer
        
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling
from transformers.file_utils import DUMMY_INPUTS, DUMMY_MASK


class TFBertSplited(PolusModel):
    
    def __init__(self, 
                 bert_layers, 
                 *args,
                 run_in_training_mode = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = bert_layers
        self.run_in_training_mode = run_in_training_mode
    
    def _efficient_attention_mask(self, x):
        # This codes mimics the transformer BERT implementation: https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py#L1057
        
        attention_mask_shape = shape_list(x)

        
        extended_attention_mask = tf.reshape(
                x, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=tf.float32)
        one_cst = tf.constant(1.0, dtype=tf.float32)
        ten_thousand_cst = tf.constant(-10000.0, dtype=tf.float32)
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        
        return extended_attention_mask
    
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], dtype=tf.float32),
                                  tf.TensorSpec([None, None], dtype=tf.int32),
                                  tf.TensorSpec([], dtype=tf.bool)],
                jit_compile=get_jit_compile())
    def call(self, hidden_states, attention_mask, training=False):
        
        attention_mask = self._efficient_attention_mask(attention_mask)
        
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states=hidden_states, 
                                         attention_mask=attention_mask,
                                         head_mask=None,
                                         output_attentions=None,
                                         encoder_hidden_states=None, 
                                         encoder_attention_mask=None, 
                                         past_key_value=None,
                                         training=(self.run_in_training_mode & training))[0]

        return TFBaseModelOutputWithPooling(last_hidden_state=hidden_states,
                                            pooler_output=hidden_states[:,0,:])

    
def split_bert_model_from_checkpoint(bert_model_checkpoint, 
                                     index_layer, 
                                     init_models=False,
                                     return_pre_bert_model=True,
                                     return_post_bert_model=True):
    
    bert_model = TFBertModel.from_pretrained(bert_model_checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = False,
                                             return_dict=True,
                                             from_pt=True)
    
    output = split_bert_model(bert_model, 
                              index_layer, 
                              init_models=init_models,
                              return_pre_bert_model=return_pre_bert_model,
                              return_post_bert_model=return_post_bert_model)
    
    if not return_pre_bert_model:
        del bert_model
    
    return output
    
def split_bert_model(bert_model, 
                     index_layer, 
                     init_models=False,
                     return_pre_bert_model=True,
                     return_post_bert_model=True):
    """
    Utility function that splits a bert model in a pre established index, given by *index_layer*,
    which results into two models. The *pre_model* that corresponds to the *bert_model* but without
    a some layers that were cut off and a *post_model* that corresponds to a *PolusModel*, which runs
    the remain bert layers.
    """
    
    assert return_pre_bert_model or return_post_bert_model # at least one must be true
    
    assert  bert_model.config.num_hidden_layers > index_layer > -bert_model.config.num_hidden_layers and index_layer!=0
    
    # create a new keras model that uses the layers previous removed bert layers
    if return_post_bert_model:
        encoder_layers = bert_model.layers[0].encoder.layer[index_layer:]
        post_model = TFBertSplited(encoder_layers)
    
    if return_pre_bert_model:
        del bert_model.layers[0].encoder.layer[index_layer:]
        bert_model.config.num_hidden_layers = len(bert_model.layers[0].encoder.layer)
    else:
        del bert_model
    
    
    
    if init_models and return_pre_bert_model:
        # run a dummy example to build post_model and check for errors
        sample = "hello, this is a sample that i want to tokenize"
        
        tokenizer = AutoTokenizer.from_pretrained(bert_model.config._name_or_path)
        
        inputs = tokenizer.encode_plus(sample,
                                           padding = "max_length",
                                           truncation = True,
                                           max_length = 50,
                                           return_attention_mask = True,
                                           return_token_type_ids = True,
                                           return_tensors = "tf",
                                          )
        hidden_states = bert_model(**inputs)["last_hidden_state"]
            
        if return_post_bert_model:
            post_model(hidden_states=hidden_states, attention_mask=inputs["attention_mask"])
    
    if return_pre_bert_model and return_post_bert_model:
        return bert_model, post_model
    if return_pre_bert_model:
        return bert_model
    if return_post_bert_model:
        return post_model