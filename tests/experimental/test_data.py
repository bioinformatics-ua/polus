from polus.experimental.data import DataLoader
import pytest
import glob
import os
import shutil

BASE_PATH_CACHE = ".polus_test_cache"
PATH_CACHE = os.path.join(BASE_PATH_CACHE, "data")

@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    
    yield # this is where the testing happens

    # Teardown : fill with any logic you want
    if os.path.exists(PATH_CACHE):
        shutil.rmtree(PATH_CACHE)

def test_dataloader_nocache():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen, 
                            use_cache = False, 
                            cache_chunk_size = 256,
                            cache_folder = PATH_CACHE)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"

def test_dataloader_mapping_f_nocache():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen, 
                            use_cache = False, 
                            cache_chunk_size = 256,
                            cache_folder = PATH_CACHE,
                            mapping_f=mapping_f)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2

    
def test_dataloader_cache():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen, 
                            use_cache = True, 
                            cache_chunk_size=256,
                            cache_folder = PATH_CACHE)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    
    # assert cache files
    cache_files = os.listdir(PATH_CACHE)
    assert f"{dataloader.cache_base_name}.index" in cache_files
    
    for data_files in dataloader.cache_index["files"]:
        assert os.path.basename(data_files) in cache_files
    
    
def test_dataloader_mapping_f_cache():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen, 
                            use_cache = False, 
                            cache_chunk_size = 256,
                            cache_folder = PATH_CACHE,
                            mapping_f=mapping_f)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    
    
def test_dataloader_mapping_f_cache_pre_shuffle():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen, 
                            use_cache = False, 
                            cache_chunk_size = 256,
                            cache_folder = PATH_CACHE,
                            mapping_f=mapping_f)
    
    dataloader.pre_shuffle()
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2