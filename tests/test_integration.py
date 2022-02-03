# integration tests as unit tests
# not the best practice
from tutorials.hpo_classifier_example import init_trainer as init_trainer
from polus.hpo import HPO_Objective, HPOContext
from polus.utils import set_random_seed


def test_mnist_classifier():
    # we dont want to mess with the other test
    set_random_seed()
    HPOContext().reset()
    
    trainer = init_trainer()
    
    trainer.train()
    
    _metrics = trainer.callbacks.shared_dict["validation"]["MNIST_Test"]["MacroF1Score"]
    
    # f1 larger than 0.9
    assert _metrics[-1]>0.9
    
def test_mnist_hpo_classifier():
    # we dont want to mess with the other test
    set_random_seed()
    HPOContext().reset()
    
    with HPO_Objective(init_trainer, "MNIST_Test", "MacroF1Score") as obj:
        print("HPO backend", obj.backend)
        
        print("Optuna config")
        for k,v in obj.optuna_cfg.items():
            print("\t",k,":",v)
        
        # change the default cfg
        obj.change_optuna_config(n_trials=5)
    
    # f1 larger than 0.9
    assert abs(obj(obj.study.best_trial) - obj.study.best_value) < 0.05
    
    
    
    
