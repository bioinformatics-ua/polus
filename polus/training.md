Module polus.training
=====================

Classes
-------

`BaseTrainer(model, optimizer, loss, metrics=[], post_process_prediction=None, post_process_grads=None)`
:   Base trainer class, this class implements an abstraction
    of all the logic needed to perform a gradient descent type
    of training.
    
    This class should not be instantiated but instead it should be
    extended by some concrete training procedure, e.g. a ClassifierTrainer
    is a trainer that implements the logic needed to train models under simple
    classification problems.
    
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

    ### Ancestors (in MRO)

    * polus.core.BaseLogger

    ### Descendants

    * polus.ir.training.EfficientDenseRetrievalTrainer
    * polus.training.ClassifierTrainer

    ### Methods

    `foward_with_grads(self, *inputs)`
    :   Describes the computations required to produce the final loss 
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

    `foward_without_grads(self, *inputs)`
    :   Foward computation that we dont want to store variables 
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

    `lr_finder(self, tf_dataset, use_lr_found=False)`
    :   Implements the lr_finder "magic trick", famoused by
        fast.ai

    `train(self, tf_dataset, epochs, callbacks=[], custom_data_transform_f=None, steps=None, learning_rate=None)`
    :

    `train_step(self, *inputs)`
    :   Describes a static computation tensorflow graph that
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

`ClassifierTrainer(model, trainable_weights=None, *args, **kwargs)`
:   Base trainer class, this class implements an abstraction
    of all the logic needed to perform a gradient descent type
    of training.
    
    This class should not be instantiated but instead it should be
    extended by some concrete training procedure, e.g. a ClassifierTrainer
    is a trainer that implements the logic needed to train models under simple
    classification problems.
    
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

    ### Ancestors (in MRO)

    * polus.training.BaseTrainer
    * polus.core.BaseLogger