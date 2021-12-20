import tensorflow as tf

from polus.core import BaseLogger, get_jit_compile


class IMetric(BaseLogger):
    def __init__(self, reduce_f=None):
        super().__init__()
        if self.__class__.__name__ == "IMetric":
            raise Exception("This is an interface that cannot be instantiated")
            
        self.name = self.__class__.__name__
        self.reduce_f = reduce_f
        
    def samples_from_batch(self, samples):
        if self.reduce_f is not None:
            samples = self.reduce_f(samples)
        
        self._samples_from_batch(samples)
    
    def _samples_from_batch(self, samples):
        raise Exception("_samples_from_batch was internally called, but is not implemented")
    
    def reset(self):
        raise Exception("clear was called, but is not implemented")
    
    def _evaluate(self):
        raise Exception("_evaluate was internally called, but is not implemented")
        
    def evaluate(self):
        measure = self._evaluate()
        if isinstance(measure, tf.Tensor):
            measure = measure.numpy()
            
        self.reset()
        return measure

    
class IConfusionMatrixTF(IMetric):
    
    def __init__(self, num_classes, reduce_f=None):
        super().__init__(reduce_f=reduce_f)
        
        if self.__class__.__name__ == "IConfusionMatrixTF":
            raise Exception("This is an interface that cannot be instantiated")
        
        self.num_classes = num_classes
        self.reset()
        
    def _samples_from_batch(self, samples):
        #print(samples["tags_int"])
        #print(samples["tags_int_pred"])
        #
        self.confusion_matrix += self._build_confusion_matrix(*samples)
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, ), dtype=tf.int32), tf.TensorSpec(shape=(None, ), dtype=tf.int32)],
                 jit_compile=get_jit_compile())
    def _build_confusion_matrix(self, y_true, y_pred):
        self.logger.debug("_build_confusion_matrix function was traced")
        #print(y_true)
        #y_true = tf.reshape(y_true, (-1,))
        #y_pred = tf.reshape(y_pred, (-1,))
        #print(y_true)
        
        return tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        
    def reset(self):
        self.confusion_matrix = tf.zeros((self.num_classes, self.num_classes), dtype=tf.int32)
        

class MacroF1Score(IConfusionMatrixTF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _evaluate(self):
        return self._tfevaluate(self.confusion_matrix)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)],
                 jit_compile=get_jit_compile())
    def _tfevaluate(self, matrix):
        self.logger.debug("MacroF1Score function was traced")
        
        tp = tf.cast(tf.linalg.tensor_diag_part(matrix), tf.float64)
        fp_tp = tf.cast(tf.math.reduce_sum(matrix, axis=-1), tf.float64)
        fn_tp = tf.cast(tf.math.reduce_sum(matrix, axis=-2), tf.float64)
        
        precision = tf.math.divide_no_nan(tp, fp_tp)
        recall = tf.math.divide_no_nan(tp, fn_tp)
        
        inv_precision = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64),  precision)
        inv_recall = tf.math.divide_no_nan(tf.constant(1, dtype=tf.float64), recall)
        
        return tf.math.reduce_mean(tf.math.divide_no_nan(tf.constant(2, dtype=tf.float64), (inv_precision+inv_recall)))
    
