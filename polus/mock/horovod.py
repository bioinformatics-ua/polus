"""
A mock file of the horovod module to just ignore its utilization
"""

def init():
    return "mock"

def local_rank():
    """
    Without a multi-gpu setup the local_rank is always 0
    """
    return 0

def size():
    return 1

def DistributedGradientTape(tape):
    return tape

def broadcast_variables(variables, root_rank=0):
    pass

def allgather_object(y):
    return [y]