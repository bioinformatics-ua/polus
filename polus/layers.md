Module polus.layers
===================

Classes
-------

`CRF(output_dim, sparse_target=True, **kwargs)`
:   #
    # Code from:
    # https://github.com/tensorflow/addons/issues/1769
    # 
    # Update by Tiago Almeida for recent versions of TF and some simplifications
    
    Args:
        output_dim (int): the number of labels to tag each temporal input.
        sparse_target (bool): whether the the ground-truth label represented in one-hot.
    Input shape:
        (batch_size, sentence length, output_dim)
    Output shape:
        (batch_size, sentence length, output_dim)

    ### Ancestors (in MRO)

    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.training.tracking.tracking.AutoTrackable
    * tensorflow.python.training.tracking.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector
    * polus.core.BaseLogger

    ### Instance variables

    `loss`
    :

    ### Methods

    `build(self, input_shape)`
    :   Creates the variables of the layer (optional, for subclass implementers).
        
        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.
        
        This is typically used to create the weights of `Layer` subclasses.
        
        Args:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

    `call(self, inputs, sequence_lengths=None, training=None, **kwargs)`
    :   This is where the layer's logic lives.
        
        Note here that `call()` method in `tf.keras` is little bit different
        from `keras` API. In `keras` API, you can pass support masking for
        layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
        method to support masking.
        
        Args:
          inputs: Input tensor, or dict/list/tuple of input tensors.
            The first positional `inputs` argument is subject to special rules:
            - `inputs` must be explicitly passed. A layer cannot have zero
              arguments, and `inputs` cannot be provided via the default value
              of a keyword argument.
            - NumPy array or Python scalar values in `inputs` get cast as tensors.
            - Keras mask metadata is only collected from `inputs`.
            - Layers are built (`build(input_shape)` method)
              using shape info from `inputs` only.
            - `input_spec` compatibility is only checked against `inputs`.
            - Mixed precision input casting is only applied to `inputs`.
              If a layer has tensor arguments in `*args` or `**kwargs`, their
              casting behavior in mixed precision should be handled manually.
            - The SavedModel input specification is generated using `inputs` only.
            - Integration with various ecosystem packages like TFMOT, TFLite,
              TF.js, etc is only supported for `inputs` and not for tensors in
              positional and keyword arguments.
          *args: Additional positional arguments. May contain tensors, although
            this is not recommended, for the reasons above.
          **kwargs: Additional keyword arguments. May contain tensors, although
            this is not recommended, for the reasons above.
            The following optional keyword arguments are reserved:
            - `training`: Boolean scalar tensor of Python boolean indicating
              whether the `call` is meant for training or inference.
            - `mask`: Boolean input mask. If the layer's `call()` method takes a
              `mask` argument, its default value will be set to the mask generated
              for `inputs` by the previous layer (if `input` did come from a layer
              that generated a corresponding mask, i.e. if it came from a Keras
              layer with masking support).
        
        Returns:
          A tensor or list/tuple of tensors.

    `compute_output_shape(self, input_shape)`
    :   Computes the output shape of the layer.
        
        If the layer has not been built, this method will call `build` on the
        layer. This assumes that the layer will later be used with inputs that
        match the input shape provided here.
        
        Args:
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.
        
        Returns:
            An input shape tuple.

    `get_config(self)`
    :   Returns the config of the layer.
        
        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).
        
        Note that `get_config()` does not guarantee to return a fresh copy of dict
        every time it is called. The callers should make a copy of the returned dict
        if they want to modify it.
        
        Returns:
            Python dictionary.

    `loss_sample_weights(self, mask_positive_classes, negative_weight)`
    :   sample_weight_vector:  list - array that contains the weight per class, which will be multiplied by all the predictions in a sequence