import tensorflow as tf
from tensorflow_addons.text import crf_log_likelihood, crf_decode

from polus import logger

class CRF(tf.keras.layers.Layer):
    """
    #
    # Code from:
    # https://github.com/tensorflow/addons/issues/1769
    # 
    # Update by Tiago Almeida for recent versions of TF and some simplifications

    """
    def __init__(self,
                 output_dim,
                 sparse_target=True,
                 mask_impossible_transitions=None,
                 **kwargs):
        """    
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)
        """
        super().__init__(**kwargs)

        self.output_dim = int(output_dim)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        self.sequence_lengths = None
        self.transitions = None
        self.mask_impossible_transitions = mask_impossible_transitions
        
        self.flatten_layer = tf.keras.layers.Flatten()

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        
        super().build(input_shape)
    
    def get_transitions(self):
        
        if self.mask_impossible_transitions is not None:
            return self.transitions * self.mask_impossible_transitions + tf.cast((tf.cast(1-self.mask_impossible_transitions, tf.int32)*-10000), tf.float32)
        
        return self.transitions
    
    def call(self, inputs, sequence_lengths=None, training=None, **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = self.flatten_layer(sequence_lengths)
        else:
            self.sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (
                tf.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.get_transitions(),
                                         self.sequence_lengths)
        
        output = tf.one_hot(viterbi_sequence, self.output_dim )
        
        return tf.keras.backend.in_train_phase(sequences, output)

    @property
    def loss(self):

        def crf_loss(y_true, y_pred):

            log_likelihood, _ = crf_log_likelihood(
                y_pred,
                tf.argmax(y_true, axis=-1, output_type=tf.dtypes.int32),
                self.sequence_lengths,
                transition_params=self.get_transitions(),
            )
            
            return tf.reduce_mean(-log_likelihood)
        return crf_loss
    
    def loss_sample_weights(self, mask_positive_classes, negative_weight):
        """
        sample_weight_vector:  list - array that contains the weight per class, which will be multiplied by all the predictions in a sequence 
        
        """
        
        def crf_loss(y_true, y_pred):

            log_likelihood, _ = crf_log_likelihood(
                y_pred,
                tf.argmax(y_true, axis=-1, output_type=tf.dtypes.int32),
                self.sequence_lengths,
                transition_params=self.get_transitions(),
            )
            
            positive_samples = y_true * mask_positive_classes
            
            negative_mask = tf.math.reduce_all(positive_samples == 0, axis=[-2,-1])
            negative_weight_per_sample = tf.cast(negative_mask, tf.float32) * negative_weight
            
            sample_weights = tf.cast(tf.math.reduce_any(positive_samples == 1, axis=[-2,-1]), tf.float32) + negative_weight_per_sample
            
            loss_per_sample = -log_likelihood * sample_weights
            
            return tf.reduce_mean(loss_per_sample)
        return crf_loss

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'supports_masking': self.supports_masking,
            'transitions': self.transitions
        }
        base_config = super(CRF, self).get_config()
        return dict(base_config, **config)