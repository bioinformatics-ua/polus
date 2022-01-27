from polus.core import Singleton, BaseLogger
from polus.training import BaseTrainer
from human_id import generate_id

import optuna 

class HPOContext(metaclass=Singleton):
    
    def __init__(self):
        self.hpo_backend = None
        
    def add_hpo_backend(self, hpo_backend):
        self.hpo_backend = hpo_backend
        
    def is_hpo_enable(self):
        return self.hpo_backend is not None
    
    def reset(self):
        self.hpo_backend = None

def parameter(real, hpo_lambda = None):
    if hpo_lambda is not None:
        ctx = HPOContext()
        if ctx.is_hpo_enable():
            return hpo_lambda(ctx.hpo_backend)
        
    return real
    
class HPO_Objective(BaseLogger):
    def __init__(self, 
                 trainer_init_function,
                 validator_name,
                 metric_name,
                 backend = "optuna",
                 metric_time = "max"):
        """
        Args:
          train_init_function (func): python function that returns a BaseTrainer
            
          val_tf_callback (polus.callbacks.ICallback): A prepered validation callback
          
          metric_name (str): name of the metric that we want to give the HPO
          
          metric_time (str): what metric value we will read, since we can record more that one
            metric per train. Availble modes [max, last]
        """
        if backend in ["optuna"]:
            self.backend = backend
        else:
            raise ValueError(f"The given backend is not supported, found {backend}")

        self.trainer_init_function = trainer_init_function
        self.validator_name = validator_name
        
        self.metric_name = metric_name
        self.metric_time = metric_time
        
        self.hpo_context = HPOContext() # this is a singleton
        
        # create the defaults for each supported backend
        if self.backend == "optuna":
            self.optuna_cfg = {
                "sampler": None,
                "pruner": None,
                "direction": "maximize",
                "study_name": generate_id(word_count=6),
                "storage": 'sqlite:///polus_optuna_hpo.db',
                "load_if_exists": True,
                "n_trials": 100,
            }
            
    def __call__(self, trial):
        # setting the backend trial
        self.hpo_context.add_hpo_backend(trial)
        
        trainer = self.trainer_init_function()
        
        # run train :)
        trainer.train()

        _metrics = trainer.callbacks.shared_dict["validation"][self.validator_name][self.metric_name]
        
        if len(_metrics)>0:
            if self.metric_time == "max":
                return max(_metrics)
            elif self.metric_time == "last":
                return _metrics[-1]
            else:
                raise ValueError(f"value used for metric_time is not supported, found {self.metric_time}")
        else:
            return -1 # fail run
        
    def change_optuna_config(self, **kwargs):
        for k,v in kwargs.items():
            self.optuna_cfg[k] = v
    
    def run(self):
        if self.backend == "optuna":
            self.study = optuna.create_study(sampler=self.optuna_cfg["sampler"],
                                             pruner=self.optuna_cfg["pruner"],
                                             direction=self.optuna_cfg["direction"],
                                             study_name=self.optuna_cfg["study_name"],
                                             storage=self.optuna_cfg["storage"],
                                             load_if_exists=self.optuna_cfg["load_if_exists"],
                                           )

            self.study.optimize(self, n_trials=self.optuna_cfg["n_trials"])
    
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.run()
        



