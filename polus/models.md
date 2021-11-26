Module polus.models
===================

Functions
---------

    
`from_config(func)`
:   

    
`load_model(file_name, change_config={}, external_module=None)`
:   

    
`resolve_activation(activation_name)`
:   

Classes
-------

`SavableModel(**kwargs)`
:   `Model` groups layers into an object with training and inference features.
    
    Args:
        inputs: The input(s) of the model: a `keras.Input` object or list of
            `keras.Input` objects.
        outputs: The output(s) of the model. See Functional API example below.
        name: String, the name of the model.
    
    There are two ways to instantiate a `Model`:
    
    1 - With the "Functional API", where you start from `Input`,
    you chain layer calls to specify the model's forward pass,
    and finally you create your model from inputs and outputs:
    
    ```python
    import tensorflow as tf
    
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    ```
    
    Note: Only dicts, lists, and tuples of input tensors are supported. Nested
    inputs are not supported (e.g. lists of list or dicts of dict).
    
    A new Functional API model can also be created by using the
    intermediate tensors. This enables you to quickly extract sub-components
    of the model.
    
    Example:
    
    ```python
    inputs = keras.Input(shape=(None, None, 3))
    processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
    conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
    pooling = keras.layers.GlobalAveragePooling2D()(conv)
    feature = keras.layers.Dense(10)(pooling)
    
    full_model = keras.Model(inputs, feature)
    backbone = keras.Model(processed, conv)
    activations = keras.Model(conv, feature)
    ```
    
    Note that the `backbone` and `activations` models are not
    created with `keras.Input` objects, but with the tensors that are originated
    from `keras.Inputs` objects. Under the hood, the layers and weights will
    be shared across these models, so that user can train the `full_model`, and
    use `backbone` or `activations` to do feature extraction.
    The inputs and outputs of the model can be nested structures of tensors as
    well, and the created models are standard Functional API models that support
    all the existing APIs.
    
    2 - By subclassing the `Model` class: in that case, you should define your
    layers in `__init__()` and you should implement the model's forward pass
    in `call()`.
    
    ```python
    import tensorflow as tf
    
    class MyModel(tf.keras.Model):
    
      def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    
      def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    model = MyModel()
    ```
    
    If you subclass `Model`, you can optionally have
    a `training` argument (boolean) in `call()`, which you can use to specify
    a different behavior in training and inference:
    
    ```python
    import tensorflow as tf
    
    class MyModel(tf.keras.Model):
    
      def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)
    
      def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
          x = self.dropout(x, training=training)
        return self.dense2(x)
    
    model = MyModel()
    ```
    
    Once the model is created, you can config the model with losses and metrics
    with `model.compile()`, train the model with `model.fit()`, or use the model
    to do prediction with `model.predict()`.

    ### Ancestors (in MRO)

    * keras.engine.training.Model
    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.training.tracking.tracking.AutoTrackable
    * tensorflow.python.training.tracking.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector
    * keras.utils.version_utils.ModelVersionSelector
    * polus.core.BaseLogger

    ### Descendants

    * polus.models.SequentialSavableModel
    * polus.ner.models.NERBertModel

    ### Methods

    `save(self, base_path='.polus_cache/saved_models', extension='')`
    :   Saves the model to Tensorflow SavedModel or a single HDF5 file.
        
        Please see `tf.keras.models.save_model` or the
        [Serialization and Saving guide](https://keras.io/guides/serialization_and_saving/)
        for details.
        
        Args:
            filepath: String, PathLike, path to SavedModel or H5 file to save the
                model.
            overwrite: Whether to silently overwrite any existing file at the
                target location, or provide the user with a manual prompt.
            include_optimizer: If True, save optimizer's state together.
            save_format: Either `'tf'` or `'h5'`, indicating whether to save the
                model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X,
                and 'h5' in TF 1.X.
            signatures: Signatures to save with the SavedModel. Applicable to the
                'tf' format only. Please see the `signatures` argument in
                `tf.saved_model.save` for details.
            options: (only applies to SavedModel format)
                `tf.saved_model.SaveOptions` object that specifies options for
                saving to SavedModel.
            save_traces: (only applies to SavedModel format) When enabled, the
                SavedModel will store the function traces for each layer. This
                can be disabled, so that only the configs of each layer are stored.
                Defaults to `True`. Disabling this will decrease serialization time
                and reduce file size, but it requires that all custom layers/models
                implement a `get_config()` method.
        
        Example:
        
        ```python
        from keras.models import load_model
        
        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
        del model  # deletes the existing model
        
        # returns a compiled model
        # identical to the previous one
        model = load_model('my_model.h5')
        ```

    `set_name(self, name)`
    :

`SequentialSavableModel(layers, **kwargs)`
:   This class is just to be compatible with the Sequential Model from keras and implements
    the same inference method from the SavableModel class
    
    Creates a `Sequential` model instance.
    
    Args:
      layers: Optional list of layers to add to the model.
      name: Optional name for the model.

    ### Ancestors (in MRO)

    * keras.engine.sequential.Sequential
    * keras.engine.functional.Functional
    * polus.models.SavableModel
    * keras.engine.training.Model
    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.training.tracking.tracking.AutoTrackable
    * tensorflow.python.training.tracking.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector
    * keras.utils.version_utils.ModelVersionSelector
    * polus.core.BaseLogger