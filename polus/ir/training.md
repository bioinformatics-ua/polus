Module polus.ir.training
========================

Classes
-------

`EfficientDenseRetrievalTrainer(model, compute_scores, k_negatives=0, trainable_weights=None, *args, **kwargs)`
:   This kind of trainer will conduct the negative sample based on the positive examples already presented
    in the batch.
    
    For instance:
    
    sample-1 -> (q_1, d_pos_1)
    sample-2 -> (q_2, d_pos_2)
    ...
    sample-N -> (q_N, d_pos_N)
    
    If we consider sample-2 as the example the d_pos_2 correspond to its positive doc, while d_pos_1;d_pos_3;...;doc_pos_N 
    can be used as negatives samples (It is important to check if the data as no conflit)
    
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