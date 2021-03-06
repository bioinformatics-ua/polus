r'''

# Polus Data API

The polus data API serves as a wrapper of the *tf.data.Dataset* API.
More precisely, we focus on the perspective of data loading and
efficient preprocessing, hence the main classes here are the DataLoaders.

### Main classes

- `polus.data.DataLoader`: Base class of data loader and automatically converts python
  generators to highly efficient *tf.data.Dataset*. Furthermore, it also supports
  GPU preprocessing which differs from *tf.data.Dataset*, since it only runs on CPU
 
- `polus.data.CachedDataLoader`: Extension of `polus.DataLoader`, adds the ability to seamlessly
  store the preprocessed samples in the disk, following a chunking mechanism. Besides the
  organizational advantage, this also enables us to implement a *pre_shuffle* mechanism
  that occurs at the chunk level, making it a much more efficient process since we can
  fully shuffle the dataset without the need to have it in memory.

- `polus.data.CachedDataLoaderwLookup`: Extension of `polus.data.CachedDataLoader`, adds the
  functionality to store arbitrary python objects jointly with the stored DataLoader.
  Note for storing python objects we use the *pickle* library.

'''


import os
import pickle
import random
import json
import types

from polus import logger
from polus.core import find_dtype_and_shapes
from polus.models import split_bert_model
import tensorflow as tf
from transformers import TFAutoModel
from transformers.modeling_tf_outputs import TFBaseModelOutputWithPooling
from functools import wraps
from timeit import default_timer as timer

from polus import PolusContext
if PolusContext().is_horovod_enabled():
    import horovod.tensorflow as hvd
else:
    import polus.mock.horovod as hvd

class DataLoader:
    """
    Base class of a data loader that builds higly efficient datasets (*tf.data.Dataset*)
    from full python generators.
    
    The python generator must return a dictionary of samples, and no other format
    is currently supported.
    
    """
    def __init__(self,
                 sample_generator,
                 magic_k=10):
        """
        
        Args:
          sample_generator (python generator): Must be a python generator that returns a
            **dictionary of samples**. Since this generator may be converted into a
            *tf.data.Dataset*, all the datatypes should be supported in TensorFlow, which is
            true for all python primitives types.
          
          magic_k (int): Integer number that defines how many samples would be executed in order to infer
            the shape and type of the data.
        
        Returns:
          None
        
        """
        super().__init__()
        
        if sample_generator is not None:
            self.name = sample_generator.__name__
        else:
            self.name = "None"
        
        self.sample_generator = sample_generator
        self.magic_k = magic_k

    def to_tfDataset(self):
        dytpes, shapes = find_dtype_and_shapes(self, 
                                               k=self.magic_k)
        
        tf_dataset = tf.data.Dataset.from_generator(lambda : self,
                                                    output_types= dytpes,
                                                    output_shapes= shapes)
        
        if PolusContext().is_horovod_enabled():
            print("----------SHARDING THE DATASET!!!!-----------")
            tf_dataset = tf_dataset.shard(num_shards = hvd.size(), index=hvd.local_rank())
            
        return tf_dataset
        
    
    def set_name(self, _name):
        self.name = _name
    
    @property
    def __name__(self):
        return f"{self.__class__.__name__}_{self.name}"

    def __iter__(self):
        # returns the function iterator
        if not isinstance(self.sample_generator, types.GeneratorType):
            sample_generator = self.sample_generator()
            if isinstance(sample_generator, types.GeneratorType):
                return sample_generator
            else:
                raise ValueError("The sample_generator that was set in the DataLoader was a function that did not return an generator, it must return a generator")
        else:
            return iter(self.sample_generator)

    def get_n_samples(self):
        """
        Returns the number of samples that the DataLoader contains,
        if the DataLoader does not know how many samples it has, then
        the DataLoader will first count the samples and then cache it
        and return the counter.
        """
        if hasattr(self, "n_samples"):
            return self.n_samples
        else:
            logger.info("this dataset does not have the number of samples in cache so it will take some time to counting")
            n_samples = 0
            for _ in self:
                n_samples += 1
                
            self.n_samples = n_samples
            
            return self.n_samples
        
class CachedDataLoader(DataLoader):
    """
    Extension of a the DataLoader class that adds the ability to seamlessly
    store the preprocessed samples in the disk, following a chunking mechanism. Besides the
    organizational advantage, this also enables us to implement a *pre_shuffle* mechanism
    that occurs at the chunk level, making it a must more efficient process since we can
    fully shuffle the dataset without the need to have its memory. of a data loader
    that builds higly efficient datasets (*tf.data.Dataset*) from full python generators.
    
    Since this class involves the storage of files on disk, if any exception is raised, the
    DataLoader will automatically clean all the created files before raising the exception
    to the user.
    
    """
    
    def __init__(self,
                 sample_generator = None,
                 clean_up_function = None,
                 cache_additional_identifier = "",
                 cache_chunk_size = 8192,
                 cache_folder=os.path.join(".polus_cache","data"),
                 cache_index = None, # this variable is used to init a CacheDataLoader with an already cached index usefull in merge
                 **kwargs):
        """
        
        Args:
          source_generator (python generator): Same as in `polus.data.DataLoader`
          
        """
        # impossible condition
        assert sample_generator is not None or cache_index is not None

        self.cache_folder = cache_folder
        self.cache_chunk_size = cache_chunk_size
        self.cache_additional_identifier = cache_additional_identifier
        self.shuffle_blocks = False
        self.clean_up_function = clean_up_function
        self.cache_index = cache_index
        
        if cache_index is not None and "cache_index_path" in cache_index:
            self.cache_index_path = cache_index["cache_index_path"]
        else:
            # this dataset probably do not have a cache_index_path, this can happen if the DataLoader was created
            # from merge of multiple DataLoader
            self.cache_index_path = None

        try:
            sample_generator = self._build_sample_generator(sample_generator)
            super().__init__(sample_generator = sample_generator, **kwargs)
        except Exception as e:
            
            if cache_index is None:
                # here we dont want to solve the exception, we just want to clean up de previously created files
                logger.info("An error has occured so all the created files will be deleted")
                self.clean()
            raise e
            
    @property
    def __name__(self):
        if self.cache_index_path is not None:
            _name = os.path.splitext(os.path.basename(self.cache_index_path))[0]
            return f"{self.__class__.__name__}_{self.name}"
        else:
            return super().__name__
    
    @classmethod
    def from_cached_index(cls, index_path):
        
        index_info = cls.read_index(index_path)
        index_info["cache_index_path"] = index_path
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
    
    def _build_sample_generator(self, sample_generator):
        
        if self.cache_index is not None:
            # we are alredy have the data needed to create the DataLoader
            # So, let's build it
            return self.__build_generator_from_index()
                 
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # get path to cache
        self.cache_base_name = self.__build_cache_base_name(sample_generator)
        self.cache_base_path = os.path.join(self.cache_folder, self.cache_base_name)
        self.cache_index_path = f"{self.cache_base_path}.index"
        
        if not os.path.exists(self.cache_index_path):
             # build a generator that reads the files from cache
            logger.info(f"DataLoader will store the samples in {self.cache_base_path}, with a max_sample per file of {self.cache_chunk_size}, this may take a while")
            # there are no history of a previous generator so we must generate the samples once and then store in cache
            generator = sample_generator
        else:
            logger.info(f"We found a compatible cache file for this DataLoader")
            generator = None
            
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
            logger.info("Starting to cache the dataset, this may take a while")
            
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
        
        if self.clean_up_function is not None:
            # TODO: make test to see if this thing works
            logger.info("Executing the clean up function after the cached dataset was created")
            self.clean_up_function()
        
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
                 
        logger.info(f"Total number of samples in dataset: {self.n_samples}")
        
        def generator():
            aux_file_index = list(range(len(self.cache_index["files"])))
            
            if self.shuffle_blocks:
                random.shuffle(aux_file_index)
                
            for file_index in aux_file_index:
                with open(self.cache_index["files"][file_index], "rb") as f:
                    for sample in pickle.load(f):
                        yield sample
                 
        return generator
        
    def __build_cache_base_name(self, sample_generator):
        name = ""
        if self.cache_additional_identifier!="":
            name = f"{self.cache_additional_identifier}_"
            
        return f"{name}_chunk{self.cache_chunk_size}_{sample_generator.__name__}"
    
    
    def pre_shuffle(self):
        """
        The order of the cached files will be readed in a random order
        """
        self.shuffle_blocks = True
        
        return self
    
    def add_lookup_data(self, lookup_object):
        """
        Converts a CachedDataLoarder to a CachedDataLoaderwLookup
        """
        cache_base_path = os.path.splitext(self.cache_index_path)[0]
        
        ## manually add lookup_data to the index
        lookup_file = f"{cache_base_path}.lookup"
        
        with open(lookup_file, "wb") as f:
            pickle.dump(lookup_object, f)
            
        return self.add_lookup_data_path(lookup_file)
            
    def add_lookup_data_path(self, lookup_data_path):
        
        assert os.path.exists(lookup_data_path)
        
        # modify the index
        self.cache_index["lookup_file"] = lookup_data_path
        
        with open(self.cache_index_path, "w") as f:
            json.dump(self.cache_index, f)
            
        return CachedDataLoaderwLookup.from_cached_index(self.cache_index_path)
    
    def deep_copy(self, path=None, suffix=None):
        
        if path is None:
            sufix = "copy" if suffix is None else suffix
            path = f"{os.path.splitext(self.cache_index_path)[0]}_{suffix}.index"
            
        with open(path, "w") as f:
            json.dump(self.cache_index, f)
        
        # update the path to the current index file
        self.cache_index_path = path

    
class CachedDataLoaderwLookup(CachedDataLoader):
    """
    Correspond to a CachedDataLoader but also keeps track of an aditional lookup file
    """
    
    def __init__(self,
                 *args,
                 lookup_data=None,
                 cache_index=None,  # this variable is used to init a CacheDataLoader with an already cached index usefull in merge
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

    
class IAccelerated_Map:
    def __init__(self):
        super().__init__()
        
        self.__name__ = self.__class__.__name__
        
        if self.__class__.__name__ == "IAccelerated_Map":
            raise Exception("This is an interface that cannot be instantiated")
            
    def build(self):
        raise ("build method was not implemented")
    

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
def build_bert_embeddings(checkpoint, bert_layer_index=None, **kwargs):
    
    bert_model = TFAutoModel.from_pretrained(checkpoint,
                                             output_attentions = False,
                                             output_hidden_states = False,
                                             return_dict=True,
                                             from_pt=True)
    
    
    
    if bert_layer_index is not None:
        pre_bert_model = split_bert_model(bert_model, bert_layer_index, return_post_bert_model=False)
        @tf.function
        def embeddings(**kwargs):
            return pre_bert_model(kwargs)
        
    else:
        @tf.function
        def embeddings(**kwargs):
            return bert_model(kwargs)
        
    return embeddings



