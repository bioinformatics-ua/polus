from polus.core import get_jit_compile, set_jit_compile, LazyTFfunction


def test_jit_compile_flag():
    
    assert get_jit_compile() == True
    set_jit_compile(False)
    assert get_jit_compile() == False
    set_jit_compile(True)
    assert get_jit_compile() == True 
    
def test_LazyTFfunction_in_class():
    
    class A:
        @LazyTFfunction
        def compute(self, x1):
            return x1 + 1
        
    x = tf.ones((10,1))
    
    obj = A()
    
    result = a.compute(x)
    assert result == (x + 1)
    
def test_LazyTFfunction_in_class_inputSig():
    
    class A:
        @LazyTFfunction(input_signature=[tf.TensorSpec([None,1], dtype=tf.float32)])
        def compute(self, x1):
            return x1 + 1
        
    x = tf.ones((10,1))
    
    obj = A()
    
    result = a.compute(x)
    assert result == (x + 1)
    

def test_LazyTFfunction_in_class_jit():
    
    class A:
        @LazyTFfunction(jit_compile=True)
        def compute(self, x1):
            return x1 + 1
        
    x = tf.ones((10,1))
    
    obj = A()
    
    result = a.compute(x)
    assert result == (x + 1)
    assert a.compute.tf_func._jit_compile == True
    
def test_LazyTFfunction_in_class_lib_jit():
    
    lib_jit = get_jit_compile()
    
    class A:
        @LazyTFfunction
        def compute(self, x1):
            return x1 + 2
        
    x = tf.ones((10,1))
    
    obj = A()
    
    result = a.compute(x)
    assert result == (x + 2)
    assert a.compute.tf_func._jit_compile == lib_jit


def test_LazyTFfunction_in_class_lib_jit_False():
    lib_jit = False
    set_jit_compile(lib_jit)
    
    class A:
        @LazyTFfunction
        def compute(self, x1):
            return x1 + 2
        
    x = tf.ones((10,1))
    
    obj = A()
    
    result = a.compute(x)
    assert result == (x + 2)
    assert a.compute.tf_func._jit_compile == lib_jit