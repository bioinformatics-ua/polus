import tensorflow as tf
import os

from polus import logger
from polus.core import get_jit_compile
from polus.callbacks import CallbackCoordinator, Profiler

from polus import PolusContext
if PolusContext().is_horovod_enabled():
    import horovod.tensorflow as hvd
else:
    import polus.mock.horovod as hvd

class BaseTrainer:
    """
    Base trainer class, this class implements an abstraction
    of all the logic needed to perform a gradient descent type
    of training.
    
    This class should not be instantiated but instead it should be
    extended by some concrete training procedure, e.g. a 
    `polus.training.ClassifierTrainer` is a trainer that implements the 
    logic needed to train models under simple classification problems.
    
    """
    def __init__(self, 
                 model,
                 optimizer, 
                 loss,
                 metrics = [],
                 post_process_logits = None,
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

          metrics: it is a list of `polus.metrics.IMetric` instances, describe what 
            measurements should be taken
            
          post_process_logits (func):
          
          post_process_grads (func):
            
        Raises:
          Exception: if this class is diractly instantiated, since this class acts only as an abstraction
        
        """
        if self.__class__.__name__ == "BaseTrainer":
            raise Exception("This is an abstraction that cannot be instantiated")
            
        super().__init__()
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.post_process_logits = post_process_logits
        self.post_process_grads = post_process_grads
        
        self.metrics = metrics
        self.early_stop = False
        
        self.train_config = {}
        
        # an internal variable that holds the number of runned steps
        self.step_counter = 0
    
        # choosing the training weights
        if not hasattr(self, "trainable_weights"):
            logger.warning((f"Since no specific trainable_weights were defined" 
                              f" during the {self.__class__.__name__} instantiation,"
                              f" the trainer will optimizer all the variables"
                              f" found on the model instance"))
            
            self.trainable_weights = model.trainable_weights
            #raise ValueError(f"{self.__class__.__name__} must define self.trainable_weights before call the super!!!")
        #self.trainable_weights = None
        self.use_horovod = PolusContext().is_horovod_enabled()
        
        if self.use_horovod:
            if hasattr(optimizer, "learning_rate"):
                _new_lr = optimizer.learning_rate.read_value()*hvd.size()
                optimizer.learning_rate.assign(_new_lr)
                logger.info(f"The learning rate was adjusted to account for the multiGPU training, local lr is {_new_lr}")
            else:
                logger.info(f"It was not possible to change the learning rate to adjusted for the multiGPU training, please make the attention to multiply the learning rate by hvd.size()")

            
        
    def __str__(self):
        return 'Trainer'
    
    
    def forward_without_grads(self, *inputs):
        """
        Foward computation that we dont want to store variables 
        and the respective intermidiate steps for the gradients
        
        Note that by default this function is optional, and it
        should be overrided in any case that we want to have
        computations that do not interfere in the error propagation
        
        Args:
          inputs (multiple tf.Tensor or dicts): These are the model inputs
            normally these would be a list of tf.Tensors or a list
            of dictionary of tf.Tensor.
            
        Returns:
          (multiple tf.Tensor or dicts): These are the inputs that are given to the
          forward_with_grads function, i.e. the inputs that are fed to the model
          that we want to train.
        """
        return inputs
    
    def forward_with_grads(self, *inputs):
        """
        Describes the computations required to produce the final loss 
        value from a predifined model or models.
        
        Note, that forward_with_grads may use self.post_process_logits
        
        This method can be further decorated with tf.function with 
        input_signature so that it can be called from lr_finder 
        and train_step without rebuilding the computation graph
        
        Args:
          inputs (multiple tf.Tensor or dicts): These are the model inputs
            normally these would be a list of tf.Tensors or a list
            of dictionary of tf.Tensor.
            
        Returns:
          (multiple tf.Tensor or dicts): inputs that are **given to the self.loss
            function**.
        
        Raises:
          NotImplementedError: if this method is not implemented
        """
        raise NotImplementedError("forward_with_grads function must be implemented in order to compute a loss value for optimization")
    
    @tf.function#()
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
          inputs (multiple tf.Tensor or dicts): list of tensors or list of dict 
          with tensors, this corresponds to the data that is
          outputed by the tf.Dataset.
          
        Returns (float): scalar error value that corresponds to
          the value outputed by the self.loss function
        
        """
        logger.debug("train_step was traced (May appear twice, more than that means that the training step is receving inputs with different shapes or dtypes)")
        
        with tf.GradientTape() as tape:           

            with tape.stop_recording():
                inputs = self.forward_without_grads(*inputs)
            
            inputs = self.forward_with_grads(*inputs)
            
            loss_value = self.loss(*inputs)
        
        tape = hvd.DistributedGradientTape(tape)
        
        # using auto-diff to get the gradients
        grads = tape.gradient(loss_value, self.trainable_weights)
        
        if self.post_process_grads is not None:
            logger.info("Post process of the gradients was added to the training loop")
            grads = self.post_process_grads(grads)
            
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss_value
    
    def lr_finder(self, tf_dataset, use_lr_found=False):
        """
        Implements the lr_finder "magic trick", famoused in
        fast.ai
        
        
        """
        pass
    
    def changing_train_config(self, **config):
        for k,v in config.items():
            self.train_config[k] = v
    
    def broadcast_init_vars(self):
        # broadcast the model variables (init) and optimizer vars 
        hvd.broadcast_variables(self.trainable_weights, root_rank=0)
        hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
        
    def train(self, 
              tf_dataset=None, 
              epochs=None,
              callbacks=[],
              train_map_f = None,
              steps = None,
              **kwargs):
        """
        Implements the main training loop, where the model is trained by
        iteratively executing the *train_step* for every *tf_dataset* sample
        during a specific amount of *epochs*.
        
        Args:
          tf_dataset (tf.data.Dataset): Training dataset that is compatible 
            with the tf.data.Dataset class, which is the case for every
            data loader in `polus.data`.
          
          epochs (int): Number of training epochs that will run during the 
            training procedure.
            
          callbacks (list of `polus.callbacks.ICallback`): List of callbacks
            that are triger during the main training loop. Use this to 
            perform operations outside each *train_step* call.
            
          train_map_f (func): A optional python function that maps the samples
            from the *tf_dataset* before the *train_step* execution. This 
            function should be implemented if we want to explicit change the 
            data samples that are fed to the *train_step* function.
            
          steps (int): Number of iterations need to empty the *tf_dataset*
            this value is optional since it is only used as user feedback, and 
            has no influence on the behaviour of the training loop.
        """
        ## argument handling
        if tf_dataset is None:
            if "tf_dataset" in self.train_config:
                tf_dataset = self.train_config["tf_dataset"]
            else:
                raise ValueError("You need to pass a training dataset to the trainer.train method")
        
        if epochs is None:
            if "epochs" in self.train_config:
                epochs = self.train_config["epochs"]
            else:
                raise ValueError("You need to pass the epochs variable to the trainer.train method")
        
        if len(callbacks)==0 and "callbacks" in self.train_config:
            callbacks = self.train_config["callbacks"]
            
        if train_map_f is None and "train_map_f" in self.train_config:
            train_map_f = self.train_config["train_map_f"]
            
        if steps is None and "steps" in self.train_config:
            steps = self.train_config["steps"]
        
        ## function logic starts here
        if steps is None:
            N_STEPS = tf.data.experimental.cardinality(tf_dataset).numpy()
        else:
            N_STEPS = steps
        
        # make backwards compatability
        if "custom_data_transform_f" in kwargs:
            train_map_f = kwargs.pop("custom_data_transform_f")

        # add polus env var
        if os.getenv("POLUS_PROFILER", 'False').lower() in ('true', '1', 't', 'y', 'yes'):
            # add the profiler callback
            callbacks.append(Profiler())
            
            
        if not isinstance(callbacks, CallbackCoordinator):
            callbacks = CallbackCoordinator(callbacks,
                                            trainer = self,
                                            epochs = epochs,
                                            steps = N_STEPS)
            
        
        self.callbacks = callbacks
        
        self.callbacks.on_train_begin()
        
        for epoch in range(epochs):
            self.callbacks.on_epoch_begin(epoch)
            
            _iter = iter(tf_dataset)
            step = 0
            
            while True:
                
            #for step, data in enumerate(tf_dataset):
                self.callbacks.on_train_batch_begin(epoch, step)
                
                data = next(_iter, None)

                if data is None:
                    break
                
                if train_map_f is not None:
                    data = train_map_f(data)
                
                if step==0 and self.use_horovod:
                    self.broadcast_init_vars()

                loss = self.train_step(*data)
                
                self.callbacks.on_train_batch_end(epoch, step, loss)
                
                ## internally increments the step counter
                self.step_counter += 1
                step +=1
                
                if self.early_stop:
                    break
                

            self.callbacks.on_epoch_end(epoch)
            
            if self.early_stop:
                break
                
        self.callbacks.on_train_end()
        
        
class ClassifierTrainer(BaseTrainer):
    """
    Standard classifier that extens the base trainer.
    
    This trainer can be use to solve simple classification 
    or regression tasks.
    """
    def __init__(self, 
                 *args,
                 **kwargs):
        """
        Args:
          args: positional arguments that are passed into the
            base class, e.g., the model variable.
            
          trainable_weights (list of tf.Variables): List of 
            trainable variables that will be optimized. Note 
            that this variables must belongs to the model to 
            be trained.
          
          kwargs: keyword arguments that are passed into the
            base class.
        """
        super().__init__(*args, **kwargs)
    
    def forward_with_grads(self, x, y):
        """
        Override of ´polus.training.BaseTrainer.forward_with_grads´
        
        Args:
          x (tf.Tensor or list(tf.Tensor) or dict(tf.Tensor)): 
            Samples, in a TensorFlow format, that are fed to the model.
            
          y (object): The true label associated with the model, this
            normally are represented as integer or float.
            
        Returns:
          y (object): The true label associated with the model, this
            normally are represented as integer or float.
            
          logits (tf.Tensor): prediction of the model over the input
            data *x*.
            
        Note that the return values are dirictly passed to the loss
        function.
        """ 
        
        if isinstance(x, dict):
            logits = self.model(**x, training=True)
        else:
            logits = self.model(x, training=True)
            
        if self.post_process_logits is not None:
            logger.info("Post process step of the logits was added to the training loop")
            logits = self.post_process_logits(logits)

        return y, logits#self.loss(y, logits)
