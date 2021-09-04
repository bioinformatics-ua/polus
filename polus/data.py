import inspect
from polus.core import BaseLogger, find_dtype_and_shapes
import tensorflow as tf
import os
def get_bert_map_function(checkpoint, bert_layer_index=-1, **kwargs): # selecting only 256
    
    bert_model = TFBertModel.from_pretrained(checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = True,
                                             return_dict=True,
                                             from_pt=True)
    
    @tf.function
    def embeddings(**kwargs):
        return bert_model(kwargs)["hidden_states"][bert_layer_index] # NONE, 512, 768

    def run_bert(data):

        data["embeddings"] = embeddings(input_ids=data["input_ids"], 
                                         token_type_ids=data["token_type_ids"],
                                         attention_mask=data["attention_mask"])

        return data
    
    return run_bert

def get_mapps_for_model(model):
    
    cfg = model.savable_config
    
    mapps = {}
    
    if "embeddings" in cfg:
        if cfg["embeddings"]["type"]=="bert":
            mapps["pre_tf_transformation"] = get_bert_map_function(**cfg["embeddings"])
            
            if "low" in cfg["model"]:
                def training(data):
                    return data["embeddings"][cfg["model"]["low"]:cfg["model"]["high"],:], tf.one_hot(data["tags_int"][cfg["model"]["low"]:cfg["model"]["high"]], 
                                                          cfg["model"]["output_classes"])
            else:
                def training(data):
                    return data["embeddings"], tf.one_hot(data["tags_int"], 
                                                          cfg["model"]["output_classes"])
            
            mapps["training"] = training
            if "low" in cfg["model"]:
                def testing(data):
                    data["embeddings"] = data["embeddings"][cfg["model"]["low"]:cfg["model"]["high"],:]
                    data["spans"] = data["spans"][cfg["model"]["low"]:cfg["model"]["high"]]
                    data["is_prediction"] = data["is_prediction"][cfg["model"]["low"]:cfg["model"]["high"]]
                    data["tags_int"] = tf.cast(data["tags_int"][cfg["model"]["low"]:cfg["model"]["high"]], tf.int32)
                    return data
            else:
                def testing(data):
                    data["tags_int"] = tf.cast(data["tags_int"], tf.int32)

                    return data
                
            mapps["testing"] = testing
            
    return mapps

short_checkpoint_names = {
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":"pubmedBertFull",
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract':"pubmedBertAbstract",
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext':"SapBert"
}

class DataLoader(BaseLogger):
    
    def __init__(self, 
                 source_generator,
                 pre_tf_transformation = None,
                 cache_name=None,
                 cache_folder=os.path.join(".polus_cache","data")):
        """
        source_generator - the source generator that feeds the data
        pre_tf_transformations - an optional transformation function that would be executed before 
                                 the tf.Dataset and is using to do heavy pre processing with the GPU
        cache - optinal variable that should specified a string path to save the output of the pre_tf_transformations function
        """
        super().__init__()
        generator = self.__build_source_generator(source_generator, pre_tf_transformation)
        
        if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
        
        self.cache_name = os.path.join(cache_folder,cache_name)
        self.n_samples = None
        dytpes, shapes = find_dtype_and_shapes(generator())
        
        self.tf_dataset = tf.data.Dataset.from_generator(generator, 
                                                         output_types= dytpes,
                                                         output_shapes= shapes)
        
        if self.cache_name is not None:
            self.tf_dataset = self.tf_dataset.cache(self.cache_name)
            
        # expose all the tf dataset methods
        for method in filter(lambda x: not x[0].startswith("_"), inspect.getmembers(self.tf_dataset, predicate=inspect.ismethod)):
            setattr(self, method[0], method[1])
    
    def __iter__(self):
        return self.tf_dataset.__iter__()
    
    @classmethod
    def for_model(cls, model, source_generators, training_map_f=None, inference_map_f=None, skip_preload=False):
        
        mapps = get_mapps_for_model(model)
        
        # build a name for the cache based on model and source_generator
        shorten_bert_name = short_checkpoint_names[model.savable_config["embeddings"]["checkpoint"]]
        bert_layer_index = model.savable_config["embeddings"]["bert_layer_index"]
        
        return_list = True
        if not isinstance(source_generators, list):
            return_list = False
            source_generators = [source_generators]
        
        to_return = []
        
        for source_generator in source_generators:
            
            
            cache_name = f"{shorten_bert_name}_layer{bert_layer_index}_{source_generator.__name__}"
            
            
            
            # build Dataloader
            data_loader = DataLoader(source_generator, 
                                 pre_tf_transformation = mapps["pre_tf_transformation"], 
                                 cache_name = cache_name)
            
            if not skip_preload:
                data_loader = data_loader.pre_build()
                
            # set train
            if training_map_f is None:
                training_ds = data_loader.map(mapps["training"])
            else:
                training_ds = data_loader.map(training_map_f)
            training_ds.data_loader = data_loader
            
            if inference_map_f is None:
                inference_ds = data_loader.map(mapps["testing"])
            else:
                inference_ds = data_loader.map(inference_map_f)    
            inference_ds.data_loader = data_loader
            
            to_return.append((training_ds, inference_ds))
            
        if return_list:
            return to_return
        else:
            return to_return[0]
    
    def pre_build(self, output_progress=True):
        if self.cache_name is not None:
            self.n_samples = 0
            
            fileList = glob.glob(self.cache_name+"*")
            if len(fileList)>0:
                self.logger.info(f"The samples will be loaded from {self.cache_name}, this should be fast")
            else:
                self.logger.info(f"The samples will be cached in {self.cache_name}, this may take some time")
            
            for i,sample in enumerate(self.tf_dataset):
                if output_progress:
                    print(f"Iteration: {i}", end="\r")
                self.n_samples += 1
                
            self.logger.info(f"This DataLoader contains {self.n_samples} samples")
        else:
            self.logger.warn(f"[This action was skipped] Since this dataloader does not use cache, does not make sence to do a pre build.")
            
        return self
    
    def clear_cache(self):
        if self.cache_name is not None:
            fileList = glob.glob(self.cache_name+"*")
            if len(fileList)>0:
                for filePath in fileList:
                    self.logger.info(f"Removing {filePath}")
                    os.remove(filePath)
                self.logger.info(f"The cache was clear")
            else:
                self.logger.info(f"The cache was already clear or the path ``{self.cache_name}´´ does not exists")
    

    
    def __build_source_generator(self, 
                                 source_generator, 
                                 pre_tf_transformations):
        if pre_tf_transformations is None:
            generator = lambda : source_generator
        else:                
            ## make an efficient pre generator using a tf.Dataset
            def generator():

                BATCH = 128

                dytpes, shapes = find_dtype_and_shapes(source_generator())

                inner_tf_dataset = tf.data.Dataset.from_generator(source_generator, 
                                                                  output_types= dytpes,
                                                                  output_shapes= shapes)\
                                                  .batch(BATCH)\
                                                  .prefetch(tf.data.experimental.AUTOTUNE)

                for data in inner_tf_dataset:

                    data = pre_tf_transformations(data)

                    ## debatching in python
                    if isinstance(data, dict):
                        key = list(data.keys())[0]
                        n_samples = data[key].shape[0] #batch size
                        for i in range(n_samples):
                            yield { k:v[i] for k, v in data.items() } 
                    elif isinstance(data, (list,tuple)):
                        n_samples = data[0].shape[0]
                        for i in range(n_samples):
                            yield [ v[i] for v in data ] 
                    else:
                        raise ValueError(f"the pre_tf_transformations function does not yield a dict nor tuple nor a list")

        return generator

