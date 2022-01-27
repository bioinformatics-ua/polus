# integration tests as unit tests
# not the best practice
from tutorials.classifier_example import init_trainer

def test_mnist_classifier():
    trainer = init_trainer()
    
    trainer.train()
    
    _metrics = trainer.callbacks.shared_dict["validation"]["MNIST_Test"]["MacroF1Score"]
    
    # f1 larger than 0.9
    assert _metrics[-1]>0.9
    
    
    
