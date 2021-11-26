Module polus.ner.metrics
========================

Classes
-------

`Accuracy(**kwargs)`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.ner.metrics.ISequentialConfusionMatrixTF
    * polus.metrics.IConfusionMatrixTF
    * polus.metrics.IMetric
    * polus.core.BaseLogger

`EntityF1(corpora)`
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

    ### Methods

    `reset(self)`
    :

`ISequentialConfusionMatrixTF(*args, **kwargs)`
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

    ### Descendants

    * polus.ner.metrics.Accuracy
    * polus.ner.metrics.MacroF1Score

    ### Methods

    `reset(self)`
    :

`MacroF1Score(**kwargs)`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.ner.metrics.ISequentialConfusionMatrixTF
    * polus.metrics.IConfusionMatrixTF
    * polus.metrics.IMetric
    * polus.core.BaseLogger