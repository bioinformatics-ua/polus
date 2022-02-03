from polus.core import get_jit_compile, set_jit_compile, Singleton
import tensorflow as tf

def test_jit_compile_flag():
    
    assert get_jit_compile() == True
    set_jit_compile(False)
    assert get_jit_compile() == False
    set_jit_compile(True)
    assert get_jit_compile() == True
    
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
    