import tensorflow as tf

from polus.core import BaseLogger, get_jit_compile
from polus.callbacks import CallbackCoordinator

class BaseTrainer(BaseLogger):
    """
    Base trainer class, this class implements an abstraction
    of all the logic needed to perform a gradient descent type
    of training.
    
    This class should not be instantiated but instead it should be
    extended by some concrete training procedure, e.g. a ClassifierTrainer
    is a trainer that implements the logic needed to train models under simple
    classification problems.
    
    """
    def __init__(self, 
                 model,
                 optimizer, 
                 loss,
                 metrics = [],
                 post_process_prediction = None,
                 post_process_grads = None):
        """
        Constructs all the necessary attributes for the training procedure

        Args:
          model (tf.keras.models.Model): it corresponds to the model instance
            that we aim to train. Furthermore the call to the model should
            behave as if is in inference mode.

          optimizer (tf.keras.optimizers.Optimizer): it corresponds to the
            optimizer instance that we want to use to update the model varibles.

          loss (func): it must be a function that returns a scalar tensor, the function
            args are free and can be defined in runtime depending on the training
            procedure

          metrics: it is a list of polus.metrics.IMetric instances, describe what 
            measurements should be taken
            
        Raises:
          Exception: if this class is diractly instantiated, since this class acts only as an abstraction
        
        """
        if self.__class__.__name__ == "BaseTrainer":
            raise Exception("This is an abstraction that cannot be instantiated")
            
        super().__init__()
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.post_process_logits = post_process_prediction
        self.post_process_grads = post_process_grads
        
        self.metrics = metrics
        self.early_stop = False
        
        # an internal variable that holds the number of runned steps
        self.step_counter = tf.Variable(0)
    
        # choosing the training weights
        if not hasattr(self, "trainable_weights"):
            self.logger.warn((f"Since no specific trainable_weights were defined" 
                              f" during the {self.__class__.__name__} instantiation,"
                              f" the trainer will optimizer all the variables"
                              f" found on the model instance"))
            
            self.trainable_weights = model.trainable_weights
            #raise ValueError(f"{self.__class__.__name__} must define self.trainable_weights before call the super!!!")
        #self.trainable_weights = None

    def __str__(self):
        return 'Trainer'
    
    
    def foward_without_grads(self, *inputs):
        """
        Foward computation that we dont want to store variables 
        and the respective intermidiate steps for the gradients
        
        Note that by default this function is optional, and it
        should be overrided in any case that we want to have
        computations that do not interfere in the error propagation
        
        Args:
          inputs (list <objects>): These are the model inputs
            normally these would be a list of tf.Tensors or a list
            of dictionary of tf.Tensor.
            
        Returns:
          (list <objects>): These are the inputs that are given to the
          forward_with_grads function, i.e. the inputs that are fed to the model
          that we want to train.
        """
        return inputs
    
    def foward_with_grads(self, *inputs):
        """
        Describes the computations required to produce the final loss 
        value from a predifined model or models.
        
        Note, that foward_with_grads may use self.post_process_logits 
        and self.loss
        
        This method can be further decorated with tf.function with 
        input_signature so that it can be called from lr_finder 
        and train_step without rebuilding the computation graph
        
        Args:
          inputs (list <objects>): These are the model inputs
            normally these would be a list of tf.Tensors or a list
            of dictionary of tf.Tensor.
            
        Returns:
          (list <objects>): inputs that are given to the self.loss
            function. 
        
        """
        raise NotImplementedError("foward_with_grads function must be implemented in order to compute a loss value for optimization")
    
    @tf.function(jit_compile=get_jit_compile())#()
    def train_step(self, *inputs):
        """
        Describes a static computation tensorflow graph that
        implements a generic training step, which encapsulates
        the forward and backpropagation computations, along side
        with the gradient estimation and respective optimization.
        
        Note: that internally this method also increments the 
        self.step_counter tf.Variable, which can then be used by
        the forward_* methods for implementing more complex logic
        
        Args:
          inputs (list <objects>): list of tensors or list of dict 
          with tensors, this corresponds to the data that is
          outputed by the tf.Dataset.
          
        Returns (float): scalar error value that corresponds to
          the value outputed by the self.loss function
        
        """
        self.logger.debug("train_step was traced (May appear twice, more than that means that that training step is receving inputs with different shapes or dtypes)")
        
        with tf.GradientTape() as tape:           

            with tape.stop_recording():
                inputs = self.foward_without_grads(*inputs)
            
            inputs = self.foward_with_grads(*inputs)
            
            loss_value = self.loss(*inputs)
        
        # using auto-diff to get the gradients
        grads = tape.gradient(loss_value, self.trainable_weights)
        
        if self.post_process_grads is not None:
            self.logger.info("Post process of the gradients was added to the training loop")
            grads = self.post_process_grads(grads)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        ## internally increments the step counter
        self.step_counter.assign_add(1)
        
        return loss_value
    
    def lr_finder(self, tf_dataset, use_lr_found=False):
        """
        Implements the lr_finder "magic trick", famoused by
        fast.ai
        
        
        """
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
        
        if trainable_weights is None:
            self.trainable_weights = model.trainable_weights
        else:
            self.trainable_weights = trainable_weights
        
        super().__init__(model, *args, **kwargs)
    
    def foward_with_grads(self, x, y):
        
        logits = self.model(x, training=True)
            
        if self.post_process_logits is not None:
            self.logger.info("Post process step of the logits was added to the training loop")
            logits = self.post_process_logits(logits)

        return y, logits#self.loss(y, logits)