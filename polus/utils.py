import tensorflow as tf
import numpy as np
import random
import json

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
    """
    Helper function to flat a dictionary of nested dictionary.
    If a key is duplicated the firsts occurence will be overrided.
    
    Adapted from https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-and-checking-how-many-key-value-pairs-are-equal
    """
    items = []
    for k, v in d.items():
        new_key = k#parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unique(iterable, key=lambda x:x):
    "Find unique items in a iterable"
    return list({ key(x):x for x in iterable }.values())

def is_jsonable(x):
    """
    https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def complex_json_serializer(data):
    _dict = {}
    for k,v in data.items():
        if isinstance(v, dict):
            _dict[k] = complex_json_serializer(v)
        elif is_jsonable(v):
            _dict[k] = v
        elif isinstance(v, tf.Tensor):
            _dict[k] = {"_class":"tensor", "dtype":v.dtype.name, "values":v.numpy().tolist()}
        else:
            raise ValueError(f"Cannot serialize {type(v)} please add a json serializer to this type of data")
        
    return _dict
        
def complex_json_deserializer(data):
    _dict = {}
    for k,v in data.items():
        if isinstance(v, dict):
            if "_class" not in v:
                _dict[k] = complex_json_deserializer(v)
            elif v["_class"]=="tensor":
                _dict[k] = tf.constant(v["values"], dtype=v["dtype"])
            else:
                _type = k["_class"]
                raise ValueError(f"Cannot deserialize {_type} please add a json deserializer to this type of data")
        else:
            _dict[k] = v

        
    return _dict