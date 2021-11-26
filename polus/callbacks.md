Module polus.callbacks
======================

Classes
-------

`Callback()`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Descendants

    * polus.callbacks.ConsoleLogCallback
    * polus.callbacks.EarlyStop
    * polus.callbacks.LossSmoothCallback
    * polus.callbacks.SaveModelCallback
    * polus.callbacks.TimerCallback
    * polus.callbacks.ValidationDataCallback
    * polus.callbacks.WandBLogCallback

    ### Methods

    `add_coordinator(self, coordinator)`
    :

`CallbackCoordinator(callbacks, trainer, epochs, steps)`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Methods

    `has_callback(self, callback_class)`
    :

    `on_epoch_begin(self, epoch)`
    :

    `on_epoch_end(self, epoch)`
    :

    `on_train_batch_begin(self, epoch, step)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

    `on_train_begin(self)`
    :

    `on_train_end(self)`
    :

`ConsoleLogCallback(log_on_train_step=False)`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.callbacks.IOutput
    * polus.core.BaseLogger

    ### Methods

    `on_epoch_begin(self, epoch)`
    :

    `on_epoch_end(self, epoch)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

    `on_train_begin(self)`
    :

    `on_train_end(self)`
    :

`EarlyStop(patience=3, use_smooth_loss=True)`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Methods

    `on_epoch_end(self, epoch)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

    `on_train_begin(self)`
    :

`ICallback()`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.core.BaseLogger

    ### Descendants

    * polus.callbacks.Callback
    * polus.callbacks.CallbackCoordinator

    ### Methods

    `on_epoch_begin(self, epoch)`
    :

    `on_epoch_end(self, epoch)`
    :

    `on_train_batch_begin(self, epoch, step)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

    `on_train_begin(self)`
    :

    `on_train_end(self)`
    :

`IOutput()`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.core.BaseLogger

    ### Descendants

    * polus.callbacks.ConsoleLogCallback
    * polus.callbacks.WandBLogCallback

    ### Methods

    `flush(self)`
    :

    `write(self, key, value)`
    :

`LossSmoothCallback(beta=0.97, output=False)`
:   Applies a smooth factor to the loss
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Methods

    `on_epoch_end(self, epoch)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

`SaveModelCallback(strategy, validation_name=None, metric_name=None, cache_folder=None, selection_dict_key=None)`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Methods

    `on_epoch_end(self, epoch)`
    :

    `on_train_end(self)`
    :

`TimerCallback()`
:   This callback only measures the time elapsed between the begin and end of a batch.
    Then the resulting measure is writed in all the OutputStreamers presented in the coordinator.
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Methods

    `on_train_batch_begin(self, epoch, step)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

`ValidationDataCallback(tf_validation, custom_inference_f=None, name=None, show_progress=False, validation_interval=1)`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.core.BaseLogger

    ### Methods

    `on_epoch_end(self, epoch)`
    :

    `on_train_begin(self)`
    :

`WandBLogCallback(project, init_args, entity=None, additional_info=None, model_config=None, model_name_prefix='')`
:   Interface
    
    Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.callbacks.Callback
    * polus.callbacks.ICallback
    * polus.callbacks.IOutput
    * polus.core.BaseLogger

    ### Methods

    `on_epoch_end(self, epoch)`
    :

    `on_train_batch_end(self, epoch, step, loss)`
    :

    `on_train_begin(self)`
    :