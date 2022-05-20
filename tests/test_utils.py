from polus.utils import flatten_dict, is_jsonable, complex_json_serializer, complex_json_deserializer, Singleton
import tensorflow as tf 
import json

    
def test_singleton_pattern():
    
    class A(metaclass=Singleton):
        def __init__(self):
            self.number = 5
            
        def increment(self):
            self.number += 1
            
    
    instance1 = A()
    assert instance1.number==5
    instance2 = A()
    instance2.increment()

    
    assert instance1 is instance2
    assert instance1.number==6
    

def test_simple_flatten_dict():
    
    a = {"b":1, "c":{"d":2},"d":{"f":{"e":3}}}
    correct_a = {'b': 1, 'e': 3, 'd': 2}
    flatten_a = flatten_dict(a)
    
    assert len(correct_a) == len(flatten_a)
    in_common = {k: correct_a[k] for k in correct_a if k in flatten_a and correct_a[k] == flatten_a[k]}
    assert len(correct_a) == len(in_common)
    
    
def test_simple_1_flatten_dict():
    
    a = {"b":1, "e":4, "c":{"d":2},"d":{"f":{"e":3}}}
    correct_a = {'b': 1, 'e': 3, 'd': 2}
    flatten_a = flatten_dict(a)
    
    assert len(correct_a) == len(flatten_a)
    in_common = {k: correct_a[k] for k in correct_a if k in flatten_a and correct_a[k] == flatten_a[k]}
    assert len(correct_a) == len(in_common)
    
def test_simple_2_flatten_dict():
    
    a = {"b":1, "c":{"d":2},"d":{"f":{"e":3}}, "e":4}
    correct_a = {'b': 1, 'e': 4, 'd': 2}
    flatten_a = flatten_dict(a)
    
    assert len(correct_a) == len(flatten_a)
    in_common = {k: correct_a[k] for k in correct_a if k in flatten_a and correct_a[k] == flatten_a[k]}
    assert len(correct_a) == len(in_common)
    
def test_is_jsonable():
    
    a = {"b":1, "c":{"d":2},"d":{"f":{"e":3}}, "e":4}
    b = {"b":tf.constant([1,2]), "c":{"d":2},"d":{"f":{"e":3}}, "e":4}
    assert is_jsonable(a)
    assert not is_jsonable(b)
    
def test_complex_json_serializer():
    
    a = {"b":1, "c":{"d":2},"d":{"f":{"e":3}}, "e":4}
    b = {"b":tf.constant([1,2]), "c":{"d":2},"d":{"f":{"e":3}}, "e":4}
    _json_a = json.dumps(complex_json_serializer(a))
    
    _json_b = json.dumps(complex_json_serializer(b))
    
    assert isinstance(_json_a, str)
    assert isinstance(_json_b, str)
    
def test_complex_json_deserializer():
    
    json_str = '{"embeddings": {"type": "bert", "checkpoint": "bla bla bla", "bert_layer_index": -1}, "model": {"sequence_length": 256, "output_classes": 4, "low": 128, "high": 384, "activation": "mish", "dropout_p": 0.4, "mask_impossible_transitions": {"_class": "tensor", "dtype": "float32", "values": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]}}}'
    
    json_data = json.loads(json_str)
    
    obj = complex_json_deserializer(json_data)
    
    assert isinstance(obj["model"]["mask_impossible_transitions"], tf.Tensor)
    assert isinstance(obj["model"]["mask_impossible_transitions"].dtype, type(tf.float32))