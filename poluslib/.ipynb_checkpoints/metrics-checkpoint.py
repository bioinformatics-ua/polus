from core import BaseLogger


class IMetric(BaseLogger):
    def __init__(self):
        super().__init__()
        if self.__class__.__name__ == "IMetric":
            raise Exception("This is an interface that cannot be instantiated")
            
        self.name = self.__class__.__name__
    
    def samples_from_batch(self, samples):
        raise Exception("samples_from_batch was called, but is not implemented")
    
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

    
