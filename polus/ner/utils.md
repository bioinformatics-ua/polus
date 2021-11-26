Module polus.ner.utils
======================

Functions
---------

    
`empty_results(counts=True)`
:   

    
`eval_list_of_entity_sets(true, pred, return_nan=True)`
:   

    
`precision_recall_f1(tp, fp, fn, return_nan=True)`
:   

Classes
-------

`BioCSequenceDecoder(corpora)`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.core.BaseLogger

    ### Class variables

    `INT2TAG`
    :

    `TAG2INT`
    :

    ### Methods

    `clear_state(self)`
    :

    `decode(self)`
    :

    `decode_from_samples(self, samples)`
    :

    `evaluate_ner(self)`
    :

    `evaluate_ner_from_sample(self, samples)`
    :

    `get_collections(self)`
    :   Return a dictionary with Collection objects.
        The first-level key is the corpus name.
        The second-level key is the group name.
        
        Each collection contains the predicted entities derived from
        the predicted samples (that have the predicted BIO tags).

    `get_collections_from_samples(self, samples)`
    :   Return a dictionary with Collection objects.
        The first-level key is the corpus name.
        The second-level key is the group name.
        
        Each collection contains the predicted entities derived from
        the predicted samples (that have the predicted BIO tags).

    `samples_from_batch(self, samples)`
    :