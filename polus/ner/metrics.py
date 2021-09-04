from collections import defaultdict
import tensorflow as tf

class EntityF1(IMetric):
    def __init__(self, corpora):
        super().__init__()
        self.sequence_decoder = SequenceDecoder(corpora)
        self.reset()
        
    def samples_from_batch(self, samples):
        self.sequence_decoder.samples_from_batch(samples)
    
    def reset(self):
        self.sequence_decoder.clear_state()
    
    def _evaluate(self):
        return self.sequence_decoder.evaluate_ner()['f1']
    
class IConfusionMatrixTF(IMetric):
    
    def __init__(self, num_classes):
        super().__init__()
        
        if self.__class__.__name__ == "IConfusionMatrixTF":
            raise Exception("This is an interface that cannot be instantiated")
        
        self.num_classes = num_classes
        self.reset()
        
    def samples_from_batch(self, samples):
        #print(samples["tags_int"])
        #print(samples["tags_int_pred"])
        self.confusion_matrix += self._build_confusion_matrix(samples["tags_int"], samples["tags_int_pred"])
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32), tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _build_confusion_matrix(self, y_true, y_pred):
        self.logger.debug("_build_confusion_matrix function was traced")
        #print(y_true)
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        
        return tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        
    def reset(self):
        self.confusion_matrix = tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32)
        
class MacroF1Score(IConfusionMatrixTF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _evaluate(self):
        return self._tfevaluate(self.confusion_matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _tfevaluate(self, matrix):
        self.logger.debug("MacroF1Score function was traced")
        
        tp = tf.linalg.tensor_diag_part(matrix)
        fp_tp = tf.math.reduce_sum(matrix, axis=-1)
        fn_tp = tf.math.reduce_sum(matrix, axis=-2)
        
        precision = tp/(fp_tp)
        recall = tp/(fn_tp)
            
        return tf.math.reduce_mean(2/((1/recall)+(1/precision)))
    
class MacroF1ScoreBI(IConfusionMatrixTF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _evaluate(self):
        return self._tfevaluate(self.confusion_matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _tfevaluate(self, matrix):
        self.logger.debug("MacroF1Score function was traced")

        matrix_true = matrix
        
        tp = tf.linalg.tensor_diag_part(matrix)[2:]
        fp_tp = tf.math.reduce_sum(matrix, axis=-1)[2:]
        fn_tp = tf.math.reduce_sum(matrix, axis=-2)[2:]
        
        precision = tp/(fp_tp)
        recall = tp/(fn_tp)
            
        return tf.math.reduce_mean(2/((1/recall)+(1/precision)))

class Accuracy(IConfusionMatrixTF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _evaluate(self):
        return self._tfevaluate(self.confusion_matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
    def _tfevaluate(self, matrix):
        self.logger.debug("Accuracy function was traced")
        
        tp = tf.math.reduce_sum(tf.linalg.tensor_diag_part(matrix))

        return tp/tf.reduce_sum(matrix)