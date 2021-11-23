import os
import pickle
import random
import json
import inspect
from polus.core import BaseLogger, find_dtype_and_shapes
import tensorflow as tf
from transformers import TFAutoModel
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling
from functools import wraps
from timeit import default_timer as timer

class IAccelerated_Map(BaseLogger):
    def __init__(self):
        super().__init__()
        
        self.__name__ = self.__class__.__name__
        
        if self.__class__.__name__ == "IAccelerated_Map":
            raise Exception("This is an interface that cannot be instantiated")
            
    def build(self):
        raise ("build method was not implemented")


class DataLoader(BaseLogger):
    
    def __init__(self, 
                 source_generator,
                 accelerated_map_f = None,
                 accelerated_map_batch = 128,
                 show_progress = False):
        """
        source_generator         - the source generator that feeds the data
        accelerated_map_f        - an optional transformation function that would be applied to 
                                   the source generator samples, this function will be executed in the best hardware that tf founds on the device
        
        """
        super().__init__()
        
        self.show_progress = show_progress
        self.accelerated_map_batch = accelerated_map_batch
        self.accelerated_map_f = accelerated_map_f
        
        self.sample_generator = self._build_sample_generator(source_generator)

        dytpes, shapes = find_dtype_and_shapes(self.sample_generator())
        
        self.tf_dataset = tf.data.Dataset.from_generator(self.sample_generator, 
                                                         output_types= dytpes,
                                                         output_shapes= shapes)
            
        # expose all the tf dataset methods
        for method in filter(lambda x: not x[0].startswith("_"), inspect.getmembers(self.tf_dataset, predicate=inspect.ismethod)):
            setattr(self, method[0], method[1])
    
    
    def _build_sample_generator(self, source_generator):
        
        # Logic to construct a generator that maybe applies the accelerated_map_f to the source_generator
        if self.accelerated_map_f is None:
            generator = source_generator
        
        else:
            # build and store the accelerated_map_f
            if isinstance(self.accelerated_map_f, IAccelerated_Map):
                # get the map function and run any heavy init only 1 time!
                self.accelerated_map_f = self.accelerated_map_f.build()
                        
            def generator():
                
                # BATCH = 128
                dytpes, shapes = find_dtype_and_shapes(source_generator())
                inner_tf_dataset = tf.data.Dataset.from_generator(source_generator, 
                                                                  output_types= dytpes,
                                                                  output_shapes= shapes)\
                                                  .batch(self.accelerated_map_batch)\
                                                  .prefetch(10)
                
                for i, data in enumerate(inner_tf_dataset):
                    
                    if self.show_progress:
                        print(f"Iteration: {i*self.accelerated_map_batch}", end="\r")
                    
                    start_bert_t = timer()
                    data = self.accelerated_map_f(data)
                    self.logger.debug(f"Time taken to run BERT {timer()-start_bert_t}")
                    ## debatching in python
                    ## TODO: this may be a bottleneck since python is slow
                    start_unbatch_t = timer()
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
                        raise ValueError(f"the accelerated_map_f function does not yield a dict nor tuple nor a list")
                    self.logger.debug(f"Time taken to unbatch {timer()-start_unbatch_t}")
            
        return generator

    def __iter__(self):
        return self.tf_dataset.__iter__()

    def get_n_samples(self):
        if hasattr(self, "n_samples"):
            return self.n_samples
        else:
            self.logger.info("this dataset does not have the number of samples in cache so it will take some time to counting")
            n_samples = 0
            for _ in self.tf_dataset:
                n_samples += 1
                
            self.n_samples = n_samples
            
            return self.n_samples
        
        
        
class CachedDataLoader(DataLoader):
    
    
    def __init__(self, 
                 source_generator = None,
                 accelerated_map_f = None,
                 accelerated_map_batch = 128,
                 show_progress = False,
                 tf_sample_map_f = None, # sample mapping function written and executed in tensorflow that is applied before the data is stored in cache
                 py_sample_map_f = None, # sample mapping function written and executed in python that is applied before the data is stored in cache
                 do_clean_up = False,
                 cache_additional_identifier = "",
                 cache_chunk_size = 8192,
                 cache_folder=os.path.join(".polus_cache","data"),
                 cache_index = None): # this variable is used to init a CacheDataLoader with an already cached index usefull in merge
        
        # impossible condition
        assert source_generator is not None or cache_index is not None

        self.cache_folder = cache_folder
        self.cache_chunk_size = cache_chunk_size
        self.cache_additional_identifier = cache_additional_identifier
        self.shuffle_blocks = False
        self.do_clean_up = do_clean_up
        self.__tf_sample_map_f = tf_sample_map_f
        self.__py_sample_map_f = py_sample_map_f
        self.cache_index = cache_index
        
        # the parent DataLoader will call _build_sample_generator that contains the logic to build the dataloader
        try:
            super().__init__(source_generator, accelerated_map_f=accelerated_map_f, show_progress=show_progress, accelerated_map_batch=accelerated_map_batch)
        except Exception as e:
            # here we dont want to solve the exception, we just want to clean up de previously created files
            self.logger.info("An error has occured so all the created files will be deleted")
            self.clean()
            raise e
            
    @classmethod
    def from_cached_index(cls, index_path):
        
        index_info = cls.read_index(index_path)
        
        return cls(cache_index=index_info)
        
    
    @classmethod
    def merge(cls, *cache_dataloaders):
        assert (len(cache_dataloaders)>1)

        index_info = {"files":[],
                      "cache_chunk_size": 0,
                      "n_samples": 0
                      }
        
        # read files
        for dl in cache_dataloaders:
            
            index = CachedDataLoader.read_index(dl.cache_index_path)
                
            index_info["n_samples"] += index["n_samples"]
            index_info["files"].extend(index["files"])
            
            # this is a bit starange, but it is possible to have different DL with diff chunk size so we will pick the larger one to define the conjunt
            index_info["cache_chunk_size"] = max(index_info["cache_chunk_size"], index["cache_chunk_size"])
            
        return cls(cache_index=index_info)
                 
    @staticmethod
    def read_index(file_path):
        with open(file_path, "r") as f:
            index = json.load(f)
        return index
    
    def _build_sample_generator(self, source_generator):
        
        if self.cache_index is not None:
            # we are alredy have the data needed to create the DataLoader
            # So, let's build it
            return self.__build_generator_from_index()
                 
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # get path to cache
        self.cache_base_name = self.__build_cache_base_name(source_generator)
        self.cache_base_path = os.path.join(self.cache_folder, self.cache_base_name)
        self.cache_index_path = f"{self.cache_base_path}.index"
        
        generator = None
        
        if not os.path.exists(self.cache_index_path):
             # build a generator that reads the files from cache
            self.logger.info(f"DataLoader will store the samples in {self.cache_base_path}, with a max_sample per file of {self.cache_chunk_size}, this may take a while")
            # there are no history of a previous generator so we must generate the samples once and then store in cache
            # normaly apply the accelerated_map_f to the source_generator
            generator = super()._build_sample_generator(source_generator)
            
            if self.__tf_sample_map_f is not None:
                tf_source_generator = generator
                
                if isinstance(self.__tf_sample_map_f, IAccelerated_Map):
                    # get the map function and run any heavy init only 1 time!
                    self.__tf_sample_map_f = self.__tf_sample_map_f.build()
                
                def generator():
                    
                    dytpes, shapes = find_dtype_and_shapes(tf_source_generator())
                    inner_tf_dataset = tf.data.Dataset.from_generator(tf_source_generator, 
                                                                      output_types= dytpes,
                                                                      output_shapes= shapes)\
                                                      .map(self.__tf_sample_map_f, num_parallel_calls=tf.data.AUTOTUNE)\
                                                      .prefetch(tf.data.AUTOTUNE)

                    for data in inner_tf_dataset:
                        yield data
            
            if self.__py_sample_map_f is not None:
                
                py_source_generator = generator
                
                if isinstance(self.__py_sample_map_f, IAccelerated_Map):
                    # get the map function and run any heavy init only 1 time!
                    self.__py_sample_map_f = self.__py_sample_map_f.build()
                
                def generator():
                    for data in py_source_generator():
                        yield self.__py_sample_map_f(data)
        else:
            self.logger.info(f"We found a compatible cache file for this DataLoader")
            
        return self._build_cache_generator(generator)
    
    def write_index_file(self, index_info):
        
        with open(self.cache_index_path, "w") as f:
            json.dump(index_info, f)
    
    def _build_cache_generator(self, generator = None):
        # first its need to store in cache the samples
        if generator is not None:
            
            index_info = {"files":[],
                          "cache_chunk_size": self.cache_chunk_size,
                          }
            
            def write_to_file(file_id, data, index):
                
                file_path = f"{self.cache_base_path}_{file_id:04}.part"

                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
                    
                index["files"].append(file_path)
            
            _temp_data = []
            _file_index = 0
            n_samples = 0
            
            # TODO: Change the tf section here so that all the tf function launched here can be destroyed
            self.logger.info("Starting to cache the dataset, this may take a while")
            
            for data in generator():
                n_samples += 1
                
                _temp_data.append(data)
                if len(_temp_data) >= self.cache_chunk_size:
                    # save data to file
                    
                    write_to_file(_file_index, _temp_data, index_info)
                    
                    # clean up 
                    _temp_data = []
                    _file_index+=1
            
            # TODO: swap to the main tf session and destroy the previous one
            
            if len(_temp_data)>0:
                # save the reminder data
                write_to_file(_file_index, _temp_data, index_info)
            
            index_info["n_samples"] = n_samples
            
            # write the index file
            self.write_index_file(index_info)
        
        if self.do_clean_up:
            # TODO: make test to see if this thing works
            self.logger.info("The current tf session will be reseted to clear the computation done by the mapping function")
            tf.keras.backend.clear_session()
        
        # read the index from file
        self.cache_index = self.__class__.read_index(self.cache_index_path)
        
        # Build the cache generator
        
        return self.__build_generator_from_index()

    
    def clean(self):
        if hasattr(self, "cache_index") and self.cache_index is not None and "files" in self.cache_index:
            for file in self.cache_index["files"]:
                if os.path.exists(file):
                    os.remove(file)
        if hasattr(self, "cache_index_path") and self.cache_index_path is not None and os.path.exists(self.cache_index_path):
            os.remove(self.cache_index_path)
                 
    def __build_generator_from_index(self):
        # set n_samples
        self.n_samples = self.cache_index["n_samples"]
        self.cache_chunk_size = self.cache_index["cache_chunk_size"]
                 
        self.logger.info(f"Total number of samples in dataset: {self.n_samples}")
        
        def generator():
            aux_file_index = list(range(len(self.cache_index["files"])))
            
            if self.shuffle_blocks:
                random.shuffle(aux_file_index)
                
            for file_index in aux_file_index:
                with open(self.cache_index["files"][file_index], "rb") as f:
                    for sample in pickle.load(f):
                        yield sample
                 
        return generator
        
    def __build_cache_base_name(self, source_generator):
        name = ""
        if self.cache_additional_identifier!="":
            name = f"{self.cache_additional_identifier}_"
        
        if self.accelerated_map_f is not None:
            name += self.accelerated_map_f.__name__
            
        if self.__tf_sample_map_f is not None:
            name += self.__tf_sample_map_f.__name__
            
        if self.__py_sample_map_f is not None:
            name += self.__py_sample_map_f.__name__
            
        return f"{name}_{source_generator.__name__}"
    
    
    def pre_shuffle(self):
        """
        The order of the cached files will be readed in a random order
        """
        self.shuffle_blocks = True
        
        return self

    
class CachedDataLoaderwLookup(CachedDataLoader):
    """
    Correspond to a CachedDataLoader but also keeps track of an aditional lookup file
    """
    
    def __init__(self, 
                 lookup_data = None,
                 cache_index=None,  # this variable is used to init a CacheDataLoader with an already cached index usefull in merge
                 *args,
                 **kwargs,
                ):
        
        if cache_index is not None and "lookup_file" in cache_index:
            with open(cache_index["lookup_file"], "rb") as f:
                lookup_data = pickle.load(f)
        
        if lookup_data is None:
            raise ValueError("Do not use CachedDataLoaderwLookup without setting a lookup_data, instead use CachedDataLoader")
        
        self.lookup_data = lookup_data
        
        # the parent DataLoader will call _build_sample_generator that contains the logic to build the dataloader
        super().__init__(*args, cache_index=cache_index, **kwargs)
        
    def get_lookup_data(self):
        return self.lookup_data
    
    @staticmethod
    def _load_lookup_data(lookup_file):
        with open(lookup_file, "rb") as f:
            return pickle.load(f)
    
    @classmethod
    def merge(cls, *cache_dataloaders):
        assert (len(cache_dataloaders)>1)

        index_info = {"files":[],
                      "cache_chunk_size": 0,
                      "n_samples": 0
                      }
        
        lookup_data = []
        
        # read files
        for dl in cache_dataloaders:
            
            index = CachedDataLoaderwLookup.read_index(dl.cache_index_path)
                
            index_info["n_samples"] += index["n_samples"]
            index_info["files"].extend(index["files"])
            
            # this is a bit starange, but it is possible to have different DL with diff chunk size so we will pick the larger one to define the conjunt
            index_info["cache_chunk_size"] = max(index_info["cache_chunk_size"], index["cache_chunk_size"])
            
            lookup_data.extend(CachedDataLoaderwLookup._load_lookup_data(index["lookup_file"]))
            
        return CachedDataLoaderwLookup(cache_index=index_info, lookup_data=lookup_data)
    
    def clean(self):

        if hasattr(self, "cache_index") and self.cache_index is not None and "lookup_file" in self.cache_index and os.path.exists(self.cache_index["lookup_file"]):
            os.remove(self.cache_index["lookup_file"])
        super().clean()
    
    def write_index_file(self, index_info):
        
        ## add lookup_data to the index
        lookup_file = f"{self.cache_base_path}.lookup"
        
        with open(lookup_file, "wb") as f:
            pickle.dump(self.lookup_data, f)
        
        index_info["lookup_file"] = lookup_file
        
        with open(self.cache_index_path, "w") as f:
            json.dump(index_info, f)
            
    def __build_generator_from_index(self):
        
        self.lookup_data = CachedDataLoaderwLookup._load_lookup_data(self.cache_index["lookup_file"])
        
        return super().__build_generator_from_index()
    
    
def access_embeddings(func):
    """
    A simple decorator function to access the embeddings property if present in the data
    """
    def function_wrapper(*args, **kwargs):
        if isinstance(kwargs, dict) and "embeddings" in kwargs:
            return func(**kwargs["embeddings"])
        else:
            return func(*args, **kwargs)
            
        
    return function_wrapper
    
@access_embeddings
def build_bert_embeddings(checkpoint, bert_layer_index=-1, **kwargs):
    
    assert bert_layer_index < 0
    
    bert_model = TFAutoModel.from_pretrained(checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = bert_layer_index!=-1,
                                             return_dict=True,
                                             from_pt=True)
    
    if bert_layer_index==-1:
        @tf.function
        def embeddings(**kwargs):
            return bert_model(kwargs)
    else:
        # use hidden_states
        @tf.function
        def embeddings(**kwargs):
            out = bert_model(kwargs)
            return TFBaseModelOutputWithPooling(last_hidden_state=out["hidden_states"][bert_layer_index],
                                                pooler_output=out["hidden_states"][bert_layer_index][:,0,:])
    
    return embeddings
