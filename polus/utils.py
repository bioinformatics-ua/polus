import tensorflow as tf
import numpy as np
import random

def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

def merge_dicts(*list_of_dicts):
    # fast merge according to https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    
    temp = dict(list_of_dicts[0], **list_of_dicts[1])
    
    for i in range(2, len(list_of_dicts)):
        temp.update(list_of_dicts[i])
        
    return temp

def flatten_dict(d):
    items = []
    for k, v in d.items():
        new_key = k#parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unique(iterable, key=lambda x:x):
    "Find unique items in a iterable"
    return list({ key(x):x for x in iterable }.values())