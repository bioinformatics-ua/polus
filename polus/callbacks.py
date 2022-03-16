from polus.core import BaseLogger, get_jit_compile
from polus.models import PolusClassifier
from polus.hpo import HPOContext

try:
    import horovod.tensorflow as hvd
except ModuleNotFoundError:
    import polus.mock.horovod as hvd

from timeit import default_timer as timer
from collections import OrderedDict, defaultdict
import tensorflow as tf
import inspect
import wandb
import numpy as np


from functools import wraps

def runs_if_root(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        if hvd.local_rank()==0:
            return method(self, *method_args, **method_kwargs)
    return _impl

class IOutput(BaseLogger):
    def __init__(self):
        super().__init__()
        if self.__class__.__name__ == "IOutputStream":
            raise Exception("This is an interface that cannot be instantiated")
            
        self.data = OrderedDict()
            
    def write(self, key, value):
        self.data[key] = value
        
    def flush(self):
        _tmp = self.data
        self.data = OrderedDict()
        return _tmp

class ICallback(BaseLogger):
    """
    Interface
    """
    def __init__(self):
        super().__init__()
        
        if self.__class__.__name__ == "ICallback":
            raise Exception("This is an interface that cannot be instantiated")
            
    def on_train_begin(self):
        pass
    
    def on_epoch_begin(self, epoch):
        pass
    
    def on_train_batch_begin(self, epoch, step):
        pass
    
    def on_train_batch_end(self, epoch, step, loss):
        pass
    
    def on_epoch_end(self, epoch):
        pass
    
    def on_train_end(self):
        pass
    
    
    
class CallbackCoordinator(ICallback):
    def __init__(self, 
                 callbacks, 
                 trainer, 
                 epochs,
                 steps):
        super().__init__()
        self.callbacks = callbacks
        self.trainer = trainer
        self.epochs = epochs
        self.steps = steps
        
        # This dict is shared across all the callbacks
        self.shared_dict = {}
        
        self.logger.info(f"{len(self.callbacks)} callbacks were registered to be used")
        
        # store the callbacks that are outputStream
        self.output_streamers = []
        
        for c in self.callbacks:
            # add cordinator to the callbacks
            c.add_coordinator(self)
            if isinstance(c, IOutput):
                self.output_streamers.append(c)
        
    
    def has_callback(self, callback_class):
        for c in self.callbacks:
            if isinstance(c, callback_class):
                return True
        return False
    
    def on_train_begin(self):
        for c in self.callbacks:
            c.on_train_begin()
    
    def on_epoch_begin(self, epoch):
        for c in self.callbacks:
            c.on_epoch_begin(epoch)
    
    def on_train_batch_begin(self, epoch, step):
        for c in self.callbacks:
            c.on_train_batch_begin(epoch, step)
    
    def on_train_batch_end(self, epoch, step, loss):
        for c in self.callbacks:
            c.on_train_batch_end(epoch, step, loss)
    
    def on_epoch_end(self, epoch):
        for c in self.callbacks:
            c.on_epoch_end(epoch)
    
    def on_train_end(self):
        for c in self.callbacks:
            c.on_train_end()
        
class Callback(ICallback):
    def __init__(self):
        super().__init__()
        self.coordinator = None # it is undifined when instantiated
        
    def add_coordinator(self, coordinator):
        self.coordinator = coordinator

class TimerCallback(Callback):
    """
    This callback only measures the time elapsed between the begin and end of a batch.
    Then the resulting measure is writed in all the OutputStreamers presented in the coordinator.
    """
    def __init__(self):
        super().__init__()
        self.start = None

    def on_train_batch_begin(self, epoch, step):
        self.start = timer()

    def on_train_batch_end(self, epoch, step, loss):
        for output in self.coordinator.output_streamers:
            output.write("time", timer()-self.start)

class LossSmoothCallback(Callback):
    """
    Applies a smooth factor to the loss
    """
    def __init__(self, beta = 0.97, output=False):
        super().__init__()
        self.beta = beta
        self.mov_avg = 0
        self.n = 0
        self.smooth_loss = 0
        self.output = output
    
    def __maybe_output(self):
        if self.output:
            for output in self.coordinator.output_streamers:
                output.write("smooth loss", self.smooth_loss)
    
    @runs_if_root
    def on_train_batch_end(self, epoch, step, loss):
        
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * loss
        self.smooth_loss = self.mov_avg / (1 - self.beta ** self.n)
        
        self.coordinator.shared_dict["smooth_loss"] = self.smooth_loss
        
        self.__maybe_output()
    
    @runs_if_root
    def on_epoch_end(self, epoch):
        self.__maybe_output()

class ValidationDataCallback(Callback):
    def __init__(self, 
                 tf_validation, 
                 custom_inference_f=None,
                 name=None,
                 show_progress=False,
                 validation_interval=1):
        super().__init__()
        self.tf_validation = tf_validation
        self.name = name
        self.custom_inference_f = custom_inference_f
        self.show_progress = show_progress
        self.validation_interval = validation_interval
    
    @runs_if_root
    def on_train_begin(self):
        if "validation" not in self.coordinator.shared_dict:
            self.coordinator.shared_dict["validation"] = {}
        
        if self.name is None:
            self.name = len(self.coordinator.shared_dict["validation"])
        
        self.coordinator.shared_dict["validation"][self.name] = {metric.name:[] for metric in self.coordinator.trainer.metrics }
    
    def get_metrics(self):
        return self.coordinator.shared_dict["validation"][self.name]
    

    def on_epoch_end(self, epoch):
        
        if not epoch%self.validation_interval:
            # only do validation at X interval

            self.logger.info(f"Running validation for {self.name} set")
            # make inference
            for step, sample in enumerate(self.tf_validation):
                if self.show_progress:
                    print(f"{step}", end="\r")

                if self.custom_inference_f is not None:
                    y = self.custom_inference_f(self.coordinator.trainer.model, sample)
                elif isinstance(sample, list) or isinstance(sample, tuple) and len(sample)==2:
                    if isinstance(self.coordinator.trainer.model, PolusClassifier):
                        y = self.coordinator.trainer.model.inference(sample[0]), sample[1]
                    else:
                        self.logger.warn(f"We default to just run the model over the validation data, since the models does not extend PolusClassifier neither a custom_inference_f was provided. This may result in erros down the line.")
                        y = self.coordinator.trainer.model(sample[0]), sample[1]
                else:
                    raise ValueError(f"Sample format outputed by the validator dataset is not supported, change to a dict or a two length tuple")
                del sample
                
                ## down below only the root should the reminder of the code
                ## THIS METHOD IS NOT EFFICIENT A MUCH MORE EFFECIENT GATHER_TO_ROOT method would be better
                all_predictions = hvd.allgather_object(y)
                
                if hvd.local_rank() == 0:
                    for pred in all_predictions:
                
                        for metric in self.coordinator.trainer.metrics:
                            metric.samples_from_batch(pred)
                            
            if hvd.local_rank() == 0:
                for metric in self.coordinator.trainer.metrics:
                    self.coordinator.shared_dict["validation"][self.name][metric.name].append(metric.evaluate())

                for output in self.coordinator.output_streamers:
                    output.write(f"Validation {self.name}", self.coordinator.shared_dict["validation"][self.name])
            
            
class SaveModelCallback(Callback):
    def __init__(self, 
                 strategy, 
                 validation_name = None,
                 metric_name = None,
                 cache_folder = None,
                 selection_dict_key=None):
        super().__init__()
        self.strategy = strategy
        self.validation_name = validation_name
        self.metric_name = metric_name
        self.cache_folder = cache_folder
        self.selection_dict_key = selection_dict_key
        
        if self.strategy not in ["every", "best", "end"]:
            self.logger.warn(f"The selected strategy ({strategy}) is not supported, so this callback will be ignored")
            
        if self.strategy == "best":
            self.best = 0
    
    @runs_if_root
    def on_epoch_end(self, epoch):
        
        if self.strategy == "best":
            _last = self.coordinator.shared_dict["validation"][self.validation_name][self.metric_name][-1]
            if isinstance(_last, dict):
                _metric = self.selection_dict_key(_last)
            else:
                _metric = _last
            
            if _metric > self.best:
                self.best = _metric
                if self.cache_folder is None:
                    self.coordinator.trainer.model.save(extension=f"_{self.validation_name}_{self.metric_name}_best")
                else:
                    self.coordinator.trainer.model.save(extension=f"_{self.validation_name}_{self.metric_name}_best", base_path=self.cache_folder)
                    
        elif self.strategy == "every":
            if self.cache_folder is None:
                self.coordinator.trainer.model.save(extension=f"_epoch_{epoch}")
            else:
                self.coordinator.trainer.model.save(extension=f"_epoch_{epoch}", base_path=self.cache_folder)
    
    @runs_if_root
    def on_train_end(self):
        if self.strategy == "end":
            if self.cache_folder is None:
                self.coordinator.trainer.model.save()
            else:
                self.coordinator.trainer.model.save(base_path=self.cache_folder)

class EarlyStop(Callback):
    def __init__(self, 
                 patience = 3, 
                 use_smooth_loss = True):
        super().__init__()
        self.current_patience = 0
        self.patience = patience
        self.last_loss = 1000
        self.use_smooth_loss = use_smooth_loss
    
    @runs_if_root
    def on_train_begin(self):
        if self.use_smooth_loss and not self.coordinator.has_callback(LossSmoothCallback):
            self.logger.warn("LossSmoothCallback was not found on the coordinator, which is a requirement to use smooth loss. Therefore this call back will use the normal loss")
            self.use_smooth_loss = False
            self.loss = []
        
    @runs_if_root
    def on_train_batch_end(self, epoch, step, loss):
        if not self.use_smooth_loss:
            self.loss.append(loss)
    
    @runs_if_root
    def on_epoch_end(self, epoch):
        if self.use_smooth_loss:
            loss = self.coordinator.shared_dict["smooth_loss"]
        else:
            loss = sum(self.loss)/len(self.loss)
            self.loss = [] # loss per epoch
        
        if np.isnan(loss):
            self.logger.info(f"The training will stop early since the loss became nan")
            self.coordinator.trainer.early_stop = True
            ctx = HPOContext()
            if ctx.is_hpo_enable():
                from optuna.exceptions import TrialPruned
                from optuna.trial import Trial
                if isinstance(ctx.hpo_backend, Trial): # optuna has backend
                    raise TrialPruned
                else:
                    raise ValueError(f"The loss was nan, but early stop does not know how to end the run with the {ctx.hpo_backend} backend")
            
        
        if self.last_loss < loss:
            self.current_patience += 1
            
        if self.current_patience > self.patience:
            self.coordinator.trainer.early_stop = True
            self.logger.info(f"The training will stop early since the loss did not improve in {self.patience} consecutive epochs")


class HPOPruneCallback(Callback):
    """
    Insperied from https://optuna.readthedocs.io/en/stable/_modules/optuna/integration/tfkeras.html#TFKerasPruningCallback
    """
    def __init__(self, validator_name, metric_name):
        super().__init__()
        
        
        
        # this can be none, if nono this callback does do anything
        self.hpo_backend = HPOContext().hpo_backend
        if self.hpo_backend is None:
            self.logger.warn(f"HPOPruneCallback was initialized however, there is no hpo context at the moment")
            
        self.validator_name = validator_name
        self.metric_name = metric_name
    
    @runs_if_root
    def on_epoch_end(self, epoch):
        
        if self.hpo_backend is not None:
            
            current_score = self.coordinator.shared_dict["validation"][self.validator_name][self.metric_name][-1]
            
            from optuna.exceptions import TrialPruned
            from optuna.trial import Trial, FrozenTrial
            
            if isinstance(self.hpo_backend, Trial): # optuna has backend
                self.hpo_backend.report(current_score, step=epoch)
                
                if self.hpo_backend.should_prune():
                    from optuna.exceptions import TrialPruned
                    message = f"Trial was pruned at epoch {epoch} with a score of {current_score}."
                    raise TrialPruned(message)
            elif isinstance(self.hpo_backend, FrozenTrial):
                ## skip the call back
                ## this means that the callback shoulden't be added in the first place
                pass
            else:
                raise ValueError(f"The current {self.hpo_backend} backend is not supported so we do not know how to prune")
        
            
class WandBLogCallback(Callback, IOutput):
    
    def __init__(self, project, init_args, entity=None, additional_info=None, model_config=None, model_name_prefix=""):
        super().__init__()
        
        self.project = project
        self.entity = entity
        self.init_args = init_args
        self.additional_info = additional_info
        self.model_config = model_config
        self.model_name_prefix = model_name_prefix

    def __flatdict(self, d):
        
        if isinstance(d, dict):
            _temp = {}
            for key, e in d.items():
                _out = self.__flatdict(e)
                if isinstance(_out, dict):
                    for k, v in _out.items():
                        _temp[f"{key} {k}"] = v
                else:
                     _temp[key] = _out
            return _temp
        elif isinstance(d, list):
            return self.__flatdict(d[-1])
        else: 
            return d

    @runs_if_root
    def on_train_begin(self):
        if self.model_config is not None:
            model_config = self.model_config
        elif hasattr(self.coordinator.trainer.model, "savable_config"):
            model_config = self.coordinator.trainer.model.savable_config
        else:
            model_config = {}
            self.logger.warning(f"The training will log information to WandB, however, the callback was unable to find the model configuration, you should explicitly set the model configuration or use @savable_model decorator")
                
        if self.entity is None:
            wandb.init(project=self.project,
                       config=model_config)
        else:
            wandb.init(project=self.project,
                       entity=self.entity,
                       config=model_config)
        
        wandb.config.update(self.init_args) 
        wandb.config.update(self.additional_info) 
        
        # add optimizer and loss
        if inspect.isfunction(self.coordinator.trainer.loss) or isinstance(self.coordinator.trainer.loss, tf.types.experimental.GenericFunction):
            _loss = {"name":self.coordinator.trainer.loss.__name__}
        else:
            _loss = self.coordinator.trainer.loss.__dict__
            _loss["name"] = self.coordinator.trainer.loss.__class__.__name__   
        
        _optimizer = self.coordinator.trainer.optimizer.__dict__
        _optimizer["name"] = self.coordinator.trainer.optimizer.__class__.__name__
        
        other_data = {"loss": _loss,
                      "optimizer": _optimizer}
        
        wandb.config.update(other_data) 
        
        # set the model name to the name generated by the wandb callback
        self.coordinator.trainer.model.set_name(wandb.run.name)#f"{self.model_name_prefix}_{wandb.run.name}")
    
    @runs_if_root
    def on_train_batch_end(self, epoch, step, loss):
        
        # store other metrics that other callback have produced
        data = self.__flatdict(self.flush())
        
        data["loss"] = loss
        
        wandb.log(data)
        
    @runs_if_root
    def on_epoch_end(self, epoch):
        # store other metrics that other callback have produced
        data = self.__flatdict(self.flush())
        
        data["epoch"] = epoch
        
        wandb.log(data)
        
class ConsoleLogCallback(Callback, IOutput):
    def __init__(self, log_on_train_step=False):
        super().__init__()
        self.log_on_train_step = log_on_train_step
        self.loss_per_epoch = defaultdict(list)
    
    def __dict2srt(self, d, sep=" - "):
        # Maybe redo this code, to be more automatic, assume a dict is open and a list prints the last element
        _tmp = ""
        if isinstance(d, dict):
            #for k,v in d.items():
                ## recursion
            _temp = []
            for key, e in d.items():
                _out = self.__dict2srt(e, ", ")
                if isinstance(e, dict) or (isinstance(e, list) and isinstance(e[0], dict)):
                    _out = f"[{_out}]"
                _temp.append(f'{key}: {_out}')
            _tmp = sep.join(_temp)

            #_tmp += f" - {k}: [{list_str}]"
        elif isinstance(d, list):
            _tmp += self.__dict2srt(d[-1], ", ")
        else: 
            _tmp += f"{d:.3f}"
                
        return _tmp
    
    @runs_if_root
    def on_train_begin(self):
        self.logger.info(f"Begin training of the model \"{self.coordinator.trainer.model.name}\" for {self.coordinator.epochs} epochs")
        jit_compiler_flag = get_jit_compile()
        self.logger.debug(f"The training step will be build with jit_compiler={jit_compiler_flag}")
        
    @runs_if_root
    def on_epoch_begin(self, epoch):
        self.logger.info(f"Begin epoch {epoch}")
    
    @runs_if_root
    def on_train_batch_end(self, epoch, step, loss):
        
        self.loss_per_epoch[epoch].append(loss)
        
        _tmp = f"{step}/{self.coordinator.steps} - loss: {loss:.3f} - "
        
        # Maybe redo this code, to be more automatic, assume a dict is open and a list prints the last element
        _tmp += self.__dict2srt(self.flush())
        
        if self.log_on_train_step:
            self.logger.info(_tmp)
        else:
            print(_tmp, end="\r")
    
    @runs_if_root
    def on_epoch_end(self, epoch):

        _len = len(self.loss_per_epoch[epoch])
        
        if _len==0:
            avg_loss = 0
        else:
            avg_loss = sum(self.loss_per_epoch[epoch])/_len
        
        _tmp = f"Average loss: {avg_loss:.3f} - "

        # print the reminder of the messages that other callbacks may have set  
        _tmp += self.__dict2srt(self.flush())
        
        self.logger.info(_tmp)
    
    @runs_if_root
    def on_train_end(self):
        self.logger.info(f"End of the training")