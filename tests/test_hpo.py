from polus.hpo import parameter, HPOContext

def test_parameter():
    
    value1 = parameter(5, lambda x: x(5))
    assert value1==5
    
    # enable HPO
    HPOContext().add_hpo_backend(lambda y: y-1) # mock some backend logic with a simple function
    
    value2 = parameter(5, lambda x: x(5))
    assert value2==4