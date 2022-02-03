from polus.core import Singleton, BaseLogger
import gc
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
                 metric_time = "max",
                 add_pruning_cb = True):
        
        """
        Args:
          train_init_function (func): python function that returns a BaseTrainer
            
          validator_name (polus.callbacks.ICallback): A prepared validation callback
          
          metric_name (str): name of the metric that we want to give the HPO
          
          metric_time (str): what metric value we will read, since we can record more that one
            metric per train. Availble modes [max, last]
        """
        super().__init__()
        if backend in ["optuna"]:
            self.backend = backend
        else:
            raise ValueError(f"The given backend is not supported, found {backend}")

        self.trainer_init_function = trainer_init_function
        self.validator_name = validator_name
        
        self.metric_name = metric_name
        self.metric_time = metric_time
        self.add_pruning_cb = add_pruning_cb
        
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
    
    def __get_validation_cb_index(self, l):
        from polus.callbacks import ValidationDataCallback
        for index, cb in enumerate(l):
            if isinstance(cb, ValidationDataCallback) and cb.name == self.validator_name:
                return index
        return -1
    
    def __add_callback_after_index(self, l, cb, index):
        return l[:index+1] + [cb] + l[index+1:]
    
    def __call__(self, trial):
        # setting the backend trial
        self.hpo_context.add_hpo_backend(trial)
        
        trainer = self.trainer_init_function()
        
        # add pruning callback to the callback list
        if self.add_pruning_cb:
            from polus.callbacks import ValidationDataCallback, HPOPruneCallback
            index = self.__get_validation_cb_index(trainer.train_config["callbacks"])
            if index != -1:
                
                trainer.train_config["callbacks"] = self.__add_callback_after_index(trainer.train_config["callbacks"],
                                                                                 HPOPruneCallback(self.validator_name, self.metric_name),
                                                                                 index)
                self.logger.info("HPO_Objective added a PruneCallback to the main training loop")
            else:
                self.logger.warn("HPO_Objective was init with add_pruning_cb, however, we could not automaticly add the pruning callback. We didn't find any Validation Callback.")
                    
        # run train :)
        trainer.train()

        _metrics = trainer.callbacks.shared_dict["validation"][self.validator_name][self.metric_name]
        
        del trainer
        gc.collect()
        
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
        # deactive the HPO mode
        self.hpo_context.reset()
        



