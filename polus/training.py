import tensorflow as tf

from polus.core import BaseLogger, get_jit_compile
from polus.callbacks import CallbackCoordinator

class BaseTrainer(BaseLogger):
    def __init__(self, 
                 optimizer, 
                 loss,
                 metrics = [],
                 post_process_prediction = None,
                 post_process_grads = None,
                 filter_weights = lambda x:x):
    
        if self.__class__.__name__ == "BaseTrainer":
            raise Exception("This is an abstraction that cannot be instantiated")
            
        super().__init__()

        self.loss = loss
        self.optimizer = optimizer
        self.post_process_logits = post_process_prediction
        self.post_process_grads = post_process_grads
        
        self.metrics = metrics
        self.early_stop = False
    
        # choosing the training weights
        if not hasattr(self, "trainable_weights"):
            raise ValueError(f"{self.__class__.__name__} must define self.trainable_weights before call the super!!!")
        #self.trainable_weights = None

    def __str__(self):
        return 'Trainer'
    
    def foward_loss_pass(self, *inputs):
        """
        Describes the computations required to produce the final loss value from a predifined model or models.
        
        Note, that foward_loss_pass may use self.post_process_logits and self.loss
        
        This method can be further decorated with tf.function with input_signature so that it can be called from lr_finder 
        and train_step without rebuilding the computation graph
        """
        raise NotImplementedError("foward_loss_pass function must be implemented in order to compute a loss value for optimization")
    
    @tf.function(jit_compile=get_jit_compile())#()
    def train_step(self, *inputs):
        self.logger.debug("train_step was traced (Should appear twice)")
        
        with tf.GradientTape() as tape:
            loss_value = self.foward_loss_pass(*inputs)
        
        # using auto-diff to get the gradients
        grads = tape.gradient(loss_value, self.trainable_weights)
        
        if self.post_process_grads is not None:
            self.logger.info("Post process of the gradients was added to the training loop")
            grads = self.post_process_grads(grads)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
        return loss_value
    
    def lr_finder(self, tf_dataset, use_lr_found=False):
        pass
    
    def train(self, 
              tf_dataset, 
              epochs,
              callbacks=[],
              custom_data_transform_f=None,
              steps = None,
              learning_rate = None):
        
        if steps is None:
            N_STEPS = tf.data.experimental.cardinality(tf_dataset).numpy()
        else:
            N_STEPS = steps
        
        if not isinstance(callbacks, CallbackCoordinator):
            callbacks = CallbackCoordinator(callbacks,
                                            trainer = self,
                                            epochs = epochs,
                                            steps = N_STEPS)
        
        callbacks.on_train_begin()
        
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            
            for step, data in enumerate(tf_dataset):
                callbacks.on_train_batch_begin(epoch, step)
                
                if custom_data_transform_f is not None:
                    data = custom_data_transform_f(data)
                
                loss = self.train_step(*data)
                
                callbacks.on_train_batch_end(epoch, step, loss)

            callbacks.on_epoch_end(epoch)
            
            if self.early_stop:
                break
                
        callbacks.on_train_end()
        
        
class ClassifierTrainer(BaseTrainer):
    
    def __init__(self, 
                 model,
                 trainable_weights = None,
                 *args,
                 **kwargs):
        
        self.model = model
        if trainable_weights is None:
            self.trainable_weights = model.trainable_weights
        else:
            self.trainable_weights = trainable_weights
        
        super().__init__(*args, **kwargs)
    
    def foward_loss_pass(self, x, y):
        
        logits = self.model(x, training=True)
            
        if self.post_process_logits is not None:
            self.logger.info("Post process step of the logits was added to the training loop")
            logits = self.post_process_logits(logits)

        return self.loss(y, logits)