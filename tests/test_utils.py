from polus.utils import flatten_dict

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
    