import os
import pickle
import random
import json
import inspect
from polus.core import BaseLogger, find_dtype_and_shapes
import tensorflow as tf
from transformers import TFBertModel
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling


class DataLoader(BaseLogger):
    
    
    def __init__(self, 
                 source_generator,
                 mapping_f = None):
        """
        source_generator - the source generator that feeds the data
        mapping_f    - an optional transformation function that would be applied to 
                           the source generator samples, ideally it can be used to do heavy pre processing on the the GPU
        """
        super().__init__()
        
        self.sample_generator = self._build_sample_generator(source_generator, mapping_f)

        dytpes, shapes = find_dtype_and_shapes(self.sample_generator())
        
        self.tf_dataset = tf.data.Dataset.from_generator(self.sample_generator, 
                                                         output_types= dytpes,
                                                         output_shapes= shapes)
            
        # expose all the tf dataset methods
        for method in filter(lambda x: not x[0].startswith("_"), inspect.getmembers(self.tf_dataset, predicate=inspect.ismethod)):
            setattr(self, method[0], method[1])
    
    
    def _build_sample_generator(self, source_generator, mapping_f):
        
        # Logic to construct a generator that maybe applies the mapping_f to the source_generator
        
        if mapping_f is None:
            generator = source_generator
        else:
            
            def generator():

                BATCH = 128
                dytpes, shapes = find_dtype_and_shapes(source_generator())
                inner_tf_dataset = tf.data.Dataset.from_generator(source_generator, 
                                                                  output_types= dytpes,
                                                                  output_shapes= shapes)\
                                                  .batch(BATCH)\
                                                  .prefetch(tf.data.experimental.AUTOTUNE)

                for data in inner_tf_dataset:

                    data = mapping_f(data)

                    ## debatching in python
                    ## TODO: this may be a bottleneck since python is slow
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
                        raise ValueError(f"the mapping_f function does not yield a dict nor tuple nor a list")
                        
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
                 source_generator,
                 mapping_f = None,
                 do_clean_up = False,
                 cache_additional_identifier = "",
                 cache_chunk_size = 8192,
                 cache_folder=os.path.join(".polus_cache","data")):


        self.cache_folder = cache_folder
        self.cache_chunk_size = cache_chunk_size
        self.cache_additional_identifier = cache_additional_identifier
        self.shuffle_blocks = False
        self.do_clean_up = do_clean_up
        
        # the parent DataLoader will call _build_sample_generator that contains the logic to build the dataloader
        super().__init__(source_generator, mapping_f=mapping_f)
            
            
    def _build_sample_generator(self, source_generator, mapping_f):

        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # get path to cache
        self.cache_base_name = self.__build_cache_base_name(source_generator, mapping_f)
        self.cache_base_path = os.path.join(self.cache_folder, self.cache_base_name)
        self.cache_index_path = f"{self.cache_base_path}.index"
        
        generator = None
        
        if not os.path.exists(self.cache_index_path):
            # there are no history of a previous generator so we must generate the samples once and then store in cache
            # normaly apply the mapping_f to the source_generator
            generator = super()._build_sample_generator(source_generator, mapping_f)

        # build a generator that reads the files from cache
        self.logger.info(f"DataLoader will store the smaples in {self.cache_base_path}, with a max_sample per file of {self.cache_chunk_size}, this may take a while")
        return self._build_cache_generator(generator)

    
    def _build_cache_generator(self, generator = None):
        # first it need to store in cache the samples
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
            
            with open(self.cache_index_path, "w") as f:
                json.dump(index_info, f)
        
        
        if self.do_clean_up:
            # TODO: make test to see if this thing works
            self.logger.info("The current tf session will be reseted to clear the computation done by the mapping function")
            tf.keras.backend.clear_session()
        
        # Build the cache generator
        
        # read the index from file
        with open(self.cache_index_path, "r") as f:
            self.cache_index = json.load(f) 
        
        
        # set n_samples
        self.n_samples = self.cache_index["n_samples"]
        
        
        def generator():
            aux_file_index = list(range(len(self.cache_index["files"])))
            
            if self.shuffle_blocks:
                random.shuffle(aux_file_index)
                
            for file_index in aux_file_index:
                with open(self.cache_index["files"][file_index], "rb") as f:
                    for sample in pickle.load(f):
                        yield sample
        
        return generator

    
    def __build_cache_base_name(self, source_generator, mapping_f):
        name = ""
        if self.cache_additional_identifier!="":
            name = f"{self.cache_additional_identifier}_"
        
        if mapping_f is not None:
            name += mapping_f.__name__
            
        return f"{name}_{source_generator.__name__}"
    
    
    def pre_shuffle(self):
        """
        The order of the cached files will be readed in a random order
        """
        self.shuffle_blocks = True
        
        return self
    
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
    
    bert_model = TFBertModel.from_pretrained(checkpoint,
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
                                                pooler_output=out["pooler_output"])
    
    return embeddings