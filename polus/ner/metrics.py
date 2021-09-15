from collections import defaultdict
from polus.ner.utils import BioCSequenceDecoder
from polus.metrics import IMetric, IConfusionMatrixTF
from polus.core import get_jit_compile

import tensorflow as tf

class EntityF1(IMetric):
    def __init__(self, corpora):
        super().__init__()
        self.sequence_decoder = BioCSequenceDecoder(corpora)
        self.reset()
        
    def _samples_from_batch(self, samples):
        self.sequence_decoder.samples_from_batch(samples)
    
    def reset(self):
        self.sequence_decoder.clear_state()
    
    def _evaluate(self):
        return self.sequence_decoder.evaluate_ner()['f1']
    
class ISequentialConfusionMatrixTF(IConfusionMatrixTF):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32)], jit_compile=get_jit_compile())
    def _build_confusion_matrix(self, y_true, y_pred):
        self.logger.debug("_build_confusion_matrix function was traced")

        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        
        return tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        
    def reset(self):
        self.confusion_matrix = tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32)
        
class MacroF1Score(ISequentialConfusionMatrixTF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _evaluate(self):
        return self._tfevaluate(self.confusion_matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)], jit_compile=get_jit_compile())
    def _tfevaluate(self, matrix):
        self.logger.debug("MacroF1Score function was traced")
        
        tp = tf.linalg.tensor_diag_part(matrix)
        fp_tp = tf.math.reduce_sum(matrix, axis=-1)
        fn_tp = tf.math.reduce_sum(matrix, axis=-2)
        
        precision = tp/(fp_tp)
        recall = tp/(fn_tp)
            
        return tf.math.reduce_mean(2/((1/recall)+(1/precision)))

class Accuracy(ISequentialConfusionMatrixTF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _evaluate(self):
        return self._tfevaluate(self.confusion_matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)], jit_compile=get_jit_compile())
    def _tfevaluate(self, matrix):
        self.logger.debug("Accuracy function was traced")
        
        tp = tf.math.reduce_sum(tf.linalg.tensor_diag_part(matrix))

        return tp/tf.reduce_sum(matrix)