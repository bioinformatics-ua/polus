Module polus.data
=================
# Polus Data API

The polus data API serves as a wrapper of the *tf.data.Dataset* API.
More precisely, we focus on the perspective of data loading and
efficient preprocessing, hence the main classes here are the DataLoaders.

### Main classes

- **DataLoader**: Base class of data loader and automatically converts python
  generators to highly efficient *tf.data.Dataset*. Furthermore, it also supports
  GPU preprocessing which differs from *tf.data.Dataset*, since it only runs on CPU
  
- **CachedDataLoader**: Extension of *polus.DataLoader*, adds the ability to seamlessly
  store the preprocessed samples in the disk, following a chunking mechanism. Besides the 
  organizational advantage, this also enables us to implement a *pre_shuffle* mechanism
  that occurs at the chunk level, making it a must more efficient process since we can
  fully shuffle the dataset without the need to have its memory. 

- **CachedDataLoaderwLookup**: Extension of *polus.CachedDataLoaderwLookup*, adds the 
  functionality to store arbitrary python objects jointly with the stored DataLoader.
  Note for storing python objects we use the *pickle* library.

Functions
---------

    
`access_embeddings(func)`
:   A simple decorator function to access the embeddings property if present in the data

    
`build_bert_embeddings(*args, **kwargs)`
:   

Classes
-------

`CachedDataLoader(source_generator=None, tf_sample_map_f=None, py_sample_map_f=None, do_clean_up=False, cache_additional_identifier='', cache_chunk_size=8192, cache_folder='.polus_cache/data', cache_index=None, **kwargs)`
:   Extension of a the DataLoader class that adds the ability to seamlessly
    store the preprocessed samples in the disk, following a chunking mechanism. Besides the 
    organizational advantage, this also enables us to implement a *pre_shuffle* mechanism
    that occurs at the chunk level, making it a must more efficient process since we can
    fully shuffle the dataset without the need to have its memory. of a data loader 
    that builds higly efficient datasets (*tf.data.Dataset*) from full python generators. 
    
    Since this class involves the storage of files on disk, if any exception is raised, the
    DataLoader will automatically clean all the created files before raising the exception
    to the user.
    
    Args:
      source_generator (python generator): Same as in `polus.data.DataLoader`

    ### Ancestors (in MRO)

    * polus.data.DataLoader
    * polus.core.BaseLogger

    ### Descendants

    * polus.data.CachedDataLoaderwLookup

    ### Static methods

    `from_cached_index(index_path)`
    :

    `merge(*cache_dataloaders)`
    :

    `read_index(file_path)`
    :

    ### Methods

    `clean(self)`
    :

    `pre_shuffle(self)`
    :   The order of the cached files will be readed in a random order

    `write_index_file(self, index_info)`
    :

`CachedDataLoaderwLookup(lookup_data=None, cache_index=None, *args, **kwargs)`
:   Correspond to a CachedDataLoader but also keeps track of an aditional lookup file
    
    Args:
      source_generator (python generator): Same as in `polus.data.DataLoader`

    ### Ancestors (in MRO)

    * polus.data.CachedDataLoader
    * polus.data.DataLoader
    * polus.core.BaseLogger

    ### Static methods

    `merge(*cache_dataloaders)`
    :

    ### Methods

    `clean(self)`
    :

    `get_lookup_data(self)`
    :

    `write_index_file(self, index_info)`
    :

`DataLoader(source_generator, accelerated_map_f=None, accelerated_map_batch=128, show_progress=False)`
:   Base class of a data loader that builds higly efficient datasets (*tf.data.Dataset*)
    from full python generators. 
    
    The python generator must return a dictionary of samples, and no other format
    is currently supported.
    
    Args:
      source_generator (python generator): Must be a python generator that returns a 
        **dictionary of samples**. Since this generator will be converted into a 
        *tf.data.Dataset*, all the datatypes must be supported in TensorFlow, which is
        true for all python primitives types.
      
      accelerated_map_f (func or polus.IAccelerated_Map): A function or interface, that
        describe a transformation to be applied to the samples from the *source_generator*.
        By default, this function will execute in the best hardware found on the host device,
        i.e., it should be used for mapping computations that should run on a GPU, e.g., 
        contextualized embeddings from BERT.
        
      accelerated_map_batch (int): The size of the batch that is fed to the *accelerated_map_f*
        recalling that when running on GPU it is more efficient to perform batch-wise operations.
        
      show_progress (boolean): If true prints the current batch that was fed to accelerated_map_f.
    
    Returns:
      None

    ### Ancestors (in MRO)

    * polus.core.BaseLogger

    ### Descendants

    * polus.data.CachedDataLoader

    ### Methods

    `get_n_samples(self)`
    :   Returns the number of samples that the DataLoader contains,
        if the DataLoader does not know how many samples it has, then
        the DataLoader will first count the samples and then cache it 
        and return the counter.

`IAccelerated_Map()`
:   Base logging class, this sets a console and file log handler to each
    instance. Meaning that each call to the logger will write to both 
    handlers. 
    
    Furthermore, the intended behaviour is to classes to extend this BaseLogger
    classe and by doing this, any class can access to the logger property
    Note that multiple instances use the same handler
    
    Main ideas from: https://www.toptal.com/python/in-depth-python-logging

    ### Ancestors (in MRO)

    * polus.core.BaseLogger

    ### Methods

    `build(self)`
    :