
import inspect
from polus.core import BaseLogger, find_dtype_and_shapes
import tensorflow as tf
import os
import pickle
import random
import json
class DataLoader(BaseLogger):
    
    def __init__(self, 
                 source_generator,
                 mapping_f = None,
                 use_cache = False,
                 cache_additional_identifier = "",
                 cache_chunk_size = 8192,
                 cache_folder=os.path.join(".polus_cache","data")):
        """
        source_generator - the source generator that feeds the data
        mapping_f    - an optional transformation function that would be applied to 
                           the source generator samples, ideally it can be used to do heavy pre processing on the the GPU
        cache - optinal variable that should specified a string path to save the output of the pre_tf_transformations function
        """
        super().__init__()
        
        self.use_cache = use_cache
        self.cache_folder = cache_folder
        self.cache_chunk_size = cache_chunk_size
        self.cache_additional_identifier = cache_additional_identifier
        self.shuffle_blocks = False
        
        self.sample_generator = self.__build_sample_generator(source_generator, mapping_f)

        dytpes, shapes = find_dtype_and_shapes(self.sample_generator())
        
        self.tf_dataset = tf.data.Dataset.from_generator(self.sample_generator, 
                                                         output_types= dytpes,
                                                         output_shapes= shapes)
            
        # expose all the tf dataset methods
        for method in filter(lambda x: not x[0].startswith("_"), inspect.getmembers(self.tf_dataset, predicate=inspect.ismethod)):
            setattr(self, method[0], method[1])
    
    def __build_sample_generator(self, source_generator, mapping_f):
        if self.use_cache:
            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)
        
            # get path to cache
            self.cache_base_name = self.__build_cache_base_name(source_generator, mapping_f)
            self.cache_base_path = os.path.join(self.cache_folder, self.cache_base_name)
            self.cache_index_path = f"{self.cache_base_path}.index"
            
            if os.path.exists(self.cache_index_path):
                # build a generator that reads the files from cache
                return self.__build_cache_generator()
            else:
                self.logger.info(f"DataLoader will cache the sample in {self.cache_base_path}")
                generator = self.__build_mapping_generator(source_generator, mapping_f)
                
                # cache the computations, this now runs in eager mode, however may be benificial to run in lazy mode
                return self.__build_cache_generator(generator)
                
        return self.__build_mapping_generator(source_generator, mapping_f)
    
    def __build_mapping_generator(self, source_generator, mapping_f):
        
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
        
        
        
    def __build_cache_generator(self, generator = None):
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
            for data in generator():
                n_samples += 1
                _temp_data.append(data)
                if len(_temp_data) >= self.cache_chunk_size:
                    # save data to file
                    write_to_file(_file_index, _temp_data, index_info)
                    
                    # clean up 
                    _temp_data = []
                    _file_index+=1
            
            if len(_temp_data)>0:
                # save the reminder data
                write_to_file(_file_index, _temp_data, index_info)
            
            index_info["n_samples"] = n_samples
            
            # write the index file
            
            with open(self.cache_index_path, "w") as f:
                json.dump(index_info, f)
            
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
    
    def __iter__(self):
        return self.tf_dataset.__iter__()
    
    def pre_shuffle(self):
        """
        If active, the order of the cached files will be readed in a random order
        """
        if not self.use_cache:
            self.logger.warning("Note that you are setting shuffle_blocks to true, however, this DataLoader does not use blocks (cache) so this option will have no effect")
        
        self.shuffle_blocks = True
        
        return self
    
    def get_n_samples(self):
        if hasattr(self, "n_samples"):
            return self.n_samples
        else:
            n_samples = 0
            for _ in self.tf_dataset:
                n_samples += 1
                
            self.n_samples = n_samples
            
            return self.n_samples