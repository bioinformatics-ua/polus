from polus.core import BaseLogger, get_jit_compile
from timeit import default_timer as timer
from collections import OrderedDict, defaultdict
import tensorflow as tf
import inspect
import wandb

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
                output.write("smooth loss",self.smooth_loss)
    
    def on_train_batch_end(self, epoch, step, loss):
        
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * loss
        self.smooth_loss = self.mov_avg / (1 - self.beta ** self.n)
        
        self.coordinator.shared_dict["smooth_loss"] = self.smooth_loss
        
        self.__maybe_output()
                
    def on_epoch_end(self, epoch):
        self.__maybe_output()

class ValidationDataCallback(Callback):
    def __init__(self, 
                 tf_validation, 
                 custom_inference_f=None,
                 name=None):
        super().__init__()
        self.tf_validation = tf_validation
        self.name = name
        self.custom_inference_f = custom_inference_f
        
    def on_train_begin(self):
        if "validation" not in self.coordinator.shared_dict:
            self.coordinator.shared_dict["validation"] = {}
        
        if self.name is None:
            self.name = len(self.coordinator.shared_dict["validation"])
        
        self.coordinator.shared_dict["validation"][self.name] = {metric.name:[] for metric in self.coordinator.trainer.metrics }
        
    def on_epoch_end(self, epoch):
        self.logger.info(f"Running validation for {self.name} set")
        # make inference

        for step, sample in enumerate(self.tf_validation):
            
            if self.custom_inference_f is not None:
                y = self.custom_inference_f(self.coordinator.trainer.model, sample)
            else:
                y = self.coordinator.trainer.model(sample)
            
            for metric in self.coordinator.trainer.metrics:
                metric.samples_from_batch(y)
                
            del sample

        for metric in self.coordinator.trainer.metrics:
            self.coordinator.shared_dict["validation"][self.name][metric.name].append(metric.evaluate())
        
        for output in self.coordinator.output_streamers:
            output.write(f"Validation {self.name}", self.coordinator.shared_dict["validation"][self.name])
            
            
class SaveModelCallback(Callback):
    def __init__(self, 
                 strategy, 
                 validation_name = None,
                 metric_name = None,
                 cache_folder = None):
        super().__init__()
        self.strategy = strategy
        self.validation_name = validation_name
        self.metric_name = metric_name
        self.cache_folder = cache_folder
        
        if self.strategy not in ["every", "best", "end"]:
            self.logger.warn(f"The selected strategy ({strategy}) is not supported, so this callback will be ignored")
            
        if self.strategy == "best":
            self.best = 0
        
    def on_epoch_end(self, epoch):
        
        if self.strategy == "best":
            if self.coordinator.shared_dict["validation"][self.validation_name][self.metric_name][-1] > self.best:
                self.best = self.coordinator.shared_dict["validation"][self.validation_name][self.metric_name][-1]
                if self.cache_folder is None:
                    self.coordinator.trainer.model.save(extension=f"_{self.validation_name}_{self.metric_name}_best")
                else:
                    self.coordinator.trainer.model.save(extension=f"_{self.validation_name}_{self.metric_name}_best", base_path=self.cache_folder)
                    
        elif self.strategy == "every":
            if self.cache_folder is None:
                self.coordinator.trainer.model.save(extension=f"_epoch_{epoch}")
            else:
                self.coordinator.trainer.model.save(extension=f"_epoch_{epoch}", base_path=self.cache_folder)
    
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
        
    def on_train_begin(self):
        if self.use_smooth_loss and not self.coordinator.has_callback(LossSmoothCallback):
            self.logger.warn("LossSmoothCallback was not found on the coordinator, which is a requirement to use smooth loss. Therefore this call back will use the normal loss")
            self.use_smooth_loss = False
            self.loss = []
            
    def on_train_batch_end(self, epoch, step, loss):
        if not self.use_smooth_loss:
            self.loss.append(loss)
            
    def on_epoch_end(self, epoch):
        if self.use_smooth_loss:
            loss = self.coordinator.shared_dict["smooth_loss"]
        else:
            loss = sum(self.loss)/len(self.loss)
            self.loss = [] # loss per epoch
            
        if self.last_loss < loss:
            self.current_patience += 1
            
        if self.current_patience > self.patience:
            self.coordinator.trainer.early_stop = True
            self.logger.info(f"The training will stop early since the loss did not improve in {self.patience} consecutive epochs")

            
class WandBLogCallback(Callback, IOutput):
    
    def __init__(self, project, init_args, entity=None, additional_info=None, model_config=None):
        super().__init__()
        
        self.project = project
        self.entity = entity
        self.init_args = init_args
        self.additional_info = additional_info
        self.model_config = model_config
     
    def __flatdict(self, d):
        _temp = {}
        for k,v in d.items():
            if isinstance(v, dict):
                for key, e in v.items():
                    _temp[f"{k} {key}"] = ( e[-1] if isinstance(e, list) else e)
            else: 
                _temp[k] = v
        return _temp
    
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
        self.coordinator.trainer.model.set_name(wandb.run.name)
        
    def on_train_batch_end(self, epoch, step, loss):
        
        # store other metrics that other callback have produced
        data = self.__flatdict(self.flush())
        
        data["loss"] = loss
        
        wandb.log(data)
            
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
    
    
    def __dict2srt(self, d):
        # Maybe redo this code, to be more automatic, assume a dict is open and a list prints the last element
        _tmp = ""
        for k,v in d.items():
            if isinstance(v, dict):
                list_str = ", ".join([ (f"{key}: {e[-1]:.3f}" if isinstance(e, list) else f"{key}: {e:.3f}") for key, e in v.items()])
                
                _tmp += f" - {k}: [{list_str}]"
            else: 
                _tmp += f" - {k}: {v:.3f}"
        return _tmp
    
    def on_train_begin(self):
        self.logger.info(f"Begin training of the model \"{self.coordinator.trainer.model.name}\" for {self.coordinator.epochs} epochs")
        jit_compiler_flag = get_jit_compile()
        self.logger.debug(f"The training step will be build with jit_compiler={jit_compiler_flag}")
        
        
    def on_epoch_begin(self, epoch):
        self.logger.info(f"Begin epoch {epoch}")
    
    def on_train_batch_end(self, epoch, step, loss):
        
        self.loss_per_epoch[epoch].append(loss)
        
        _tmp = f"{step}/{self.coordinator.steps} - loss: {loss:.3f}"
        
        # Maybe redo this code, to be more automatic, assume a dict is open and a list prints the last element
        _tmp += self.__dict2srt(self.flush())
        
        if self.log_on_train_step:
            self.logger.info(_tmp)
        else:
            print(_tmp, end="\r")
    
    def on_epoch_end(self, epoch):
        
        _tmp = f"Average loss: {sum(self.loss_per_epoch[epoch])/len(self.loss_per_epoch[epoch]):.3f}"

        # print the reminder of the messages that other callbacks may have set       
        _tmp += self.__dict2srt(self.flush())
        
        self.logger.info(_tmp)
    
    def on_train_end(self):
        self.logger.info(f"End of the training")