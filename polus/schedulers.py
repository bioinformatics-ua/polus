import tensorflow as tf
from transformers.optimization_tf import WarmUp, AdamWeightDecay


def warmup_scheduler(num_train_steps, 
                     max_lr, 
                     warmup_percentage=0.1, 
                     end_lr=1e-7):

    num_warmup_steps = int(num_train_steps*0.1)
    
    lr_linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=max_lr,
        decay_steps=num_train_steps - num_warmup_steps,
        end_learning_rate=1e-7,
        power=1,
    )

    return WarmUp(
        initial_learning_rate=max_lr,
        decay_schedule_fn=lr_linear_decay,
        warmup_steps=num_warmup_steps,
    )
    
    
    
