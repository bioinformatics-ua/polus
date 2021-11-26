from polus.core import get_jit_compile, set_jit_compile


def test_jit_compile_flag():
    
    assert get_jit_compile() == True
    set_jit_compile(False)
    assert get_jit_compile() == False
    set_jit_compile(True)
    assert get_jit_compile() == True 
    