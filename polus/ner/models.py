import tensorflow as tf

from polus.layers import CRF
from polus.models import SavableModel

class NERBertModel(SavableModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
    
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32)])
    def inference(self, x):
        self.logger.debug("Inference function was traced")
        return tf.argmax(self(x), axis=-1, output_type=tf.dtypes.int32)
    
    
class SequentialNERBertModel(tf.keras.Sequential, NERBertModel):
    """
    This class is just to be compatible with the Sequential Model from keras and implements
    the same inference method from the Classifier Model class
    """
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

        
def resolve_activation(activation_name):
    if activation_name=="mish":
        return tfa.activations.mish
    else:
        return activation_name
    
    
@savable_model
def baselineNER_MLP_CRF(sequence_length=256, 
                        output_classes = 3, 
                        hidden_space = 128,
                        activation=tf.keras.activations.swish, 
                        **kwargs):
    
    activation = resolve_activation(activation)
    
    crf_layer = CRF(output_classes)
    
    model = SequentialNERBertModel([
        tf.keras.layers.Dense(hidden_space, input_shape=(sequence_length, 768), activation=activation),
        tf.keras.layers.Dense(output_classes),
        crf_layer
    ])
    
    model.loss = crf_layer.loss
    model.loss_sample_weights = crf_layer.loss_sample_weights
    
    return model