Module polus.metrics
====================

Classes
-------

`IConfusionMatrixTF(num_classes, reduce_f=None)`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.metrics.IMetric
    * polus.core.BaseLogger

    ### Descendants

    * polus.metrics.MacroF1Score
    * polus.ner.metrics.ISequentialConfusionMatrixTF

    ### Methods

    `reset(self)`
    :

`IMetric(reduce_f=None)`
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

    * polus.metrics.IConfusionMatrixTF
    * polus.ner.metrics.EntityF1

    ### Methods

    `evaluate(self)`
    :

    `reset(self)`
    :

    `samples_from_batch(self, samples)`
    :

`MacroF1Score(*args, **kwargs)`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.metrics.IConfusionMatrixTF
    * polus.metrics.IMetric
    * polus.core.BaseLogger