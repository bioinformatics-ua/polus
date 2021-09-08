import tensorflow as tf

from polus.core import BaseLogger, get_jit_compile
from polus.callbacks import CallbackCoordinator

class Trainer(BaseLogger):
    def __init__(self, 
                 model, 
                 optimizer, 
                 loss,
                 metrics = [],
                 post_process_prediction = None,
                 post_process_grads = None,
                 filter_weights = lambda x:x):
        super().__init__()
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.post_process_logits = post_process_prediction
        self.post_process_grads = post_process_grads
        
        self.metrics = metrics
        self.early_stop = False
        
        # choosing the training weights
        self.trainable_weights = filter_weights(self.model.trainable_weights)
        
    def __str__(self):
        return 'Trainer'
    
    @tf.function(jit_compile=get_jit_compile())#()
    def train_step(self, x, y):
        self.logger.debug("train_step was traced (Should appear twice)")
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            
            if self.post_process_logits is not None:
                self.logger.info("Post process step of the logits was added to the training loop")
                logits = self.post_process_logits(logits)

            loss_value = self.loss(y, logits)
        
        grads = tape.gradient(loss_value, self.trainable_weights)
        
        # using auto-diff to get the gradients
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