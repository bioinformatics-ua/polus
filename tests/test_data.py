from polus.data import DataLoader, CachedDataLoader, CachedDataLoaderwLookup, build_bert_embeddings
import pytest
import glob
import os
import shutil

from transformers import BertTokenizerFast, TFBertModel
import tensorflow as tf

from tests.utils import vector_equals

BERT_CHECKPOINT = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
BASE_PATH_CACHE = ".polus_test_cache"
PATH_CACHE = os.path.join(BASE_PATH_CACHE, "data")
N_SAMPLES = 1000

@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    
    yield # this is where the testing happens

    # Teardown : fill with any logic you want
    if os.path.exists(PATH_CACHE):
        shutil.rmtree(PATH_CACHE)

def test_dataloader():
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen)
    
    samples_cumulative_equality = True
    n_samples = 0
    for sample in dataloader:
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"] == n_samples)
        n_samples += 1
    
    assert samples_cumulative_equality
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    
def test_dataloader_tfDataset():
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen)
    
    samples_cumulative_equality = True
    n_samples = 0
    for sample in dataloader.to_tfDataset():
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"].numpy() == n_samples)
        n_samples += 1
    
    assert samples_cumulative_equality
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"].numpy() == N_SAMPLES-1
    assert sample["text"].numpy().decode() == "dummy"
    
def test_cachedDataloader():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE)
    
    samples_cumulative_equality = True
    n_samples = 0
    for sample in dataloader:
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"] == n_samples)
        n_samples += 1
    
    assert samples_cumulative_equality
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    
    # assert cache files
    cache_files = os.listdir(PATH_CACHE)
    assert f"{dataloader.cache_base_name}.index" in cache_files
    
    for data_files in dataloader.cache_index["files"]:
        assert os.path.basename(data_files) in cache_files
        
    
def test_cachedDataloader_tfDataset():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE)
    
    samples_cumulative_equality = True
    n_samples = 0
    for sample in dataloader.to_tfDataset():
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"].numpy() == n_samples)
        n_samples += 1
    
    assert samples_cumulative_equality
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"].numpy() == N_SAMPLES-1
    assert sample["text"].numpy().decode() == "dummy"
    
    # assert cache files
    cache_files = os.listdir(PATH_CACHE)
    assert f"{dataloader.cache_base_name}.index" in cache_files
    
    for data_files in dataloader.cache_index["files"]:
        assert os.path.basename(data_files) in cache_files

def test_cachedDataloaderwLookup():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    data = {"a":1, "b":2, "c":3}    
        
    dataloader = CachedDataLoaderwLookup(source_gen, 
                                         lookup_data = data,
                                         cache_chunk_size = 256,
                                         cache_folder = PATH_CACHE)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    assert dataloader.get_lookup_data()["c"] == data["c"]
    
    # assert cache files
    cache_files = os.listdir(PATH_CACHE)
    assert f"{dataloader.cache_base_name}.index" in cache_files
    assert f"{dataloader.cache_base_name}.lookup" in cache_files
    
    for data_files in dataloader.cache_index["files"]:
        assert os.path.basename(data_files) in cache_files
        
    
def test_cachedDataloader_convert():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE)
    
    data = {"a":1, "b":2, "c":3}    
    dataloader = dataloader.add_lookup_data(data)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert isinstance(dataloader, CachedDataLoaderwLookup)
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    assert dataloader.get_lookup_data()["c"] == data["c"]
    
    # assert cache files
    cache_files = os.listdir(PATH_CACHE)
    
    base_name = os.path.splitext(os.path.basename(dataloader.cache_index_path))[0]
    
    assert f"{base_name}.index" in cache_files
    assert f"{base_name}.lookup" in cache_files
    
    for data_files in dataloader.cache_index["files"]:
        assert os.path.basename(data_files) in cache_files
    

def test_cachedDataloader_custom_gens():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    def new_gen():
        for s in source_gen():
            yield mapping_f(s)
            
    dataloader = CachedDataLoader(new_gen, 
                                  cache_chunk_size = 16,
                                  cache_folder = PATH_CACHE)
    
    samples_cumulative_equality = True
    n_samples = 0
    for sample in dataloader:
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"] == n_samples and sample["id"]*2==sample["new_entry"])
        n_samples += 1
    
    assert samples_cumulative_equality # this test ensures order
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    
def test_cachedDataloader_pre_shuffle():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    def new_gen():
        for s in source_gen():
            yield mapping_f(s)
            
    dataloader = CachedDataLoader(new_gen, 
                                  cache_chunk_size = 16,
                                  cache_folder = PATH_CACHE)
    
    dataloader.pre_shuffle()
    
    samples_cumulative_equality = True
    n_samples = 0
    for sample in dataloader:
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"] == n_samples and sample["id"]*2==sample["new_entry"])
        n_samples += 1
    
    assert not samples_cumulative_equality # this test ensures order
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    
    
def test_cachedDataloader_MERGE_pre_shuffle():

        
    def source_gen_1():
        for i in range(1000):
            yield {"id":i, "text":"dummy"}
    
    def source_gen_2():
        for i in range(500):
            yield {"id":i, "text":"dummy2"}
            
    def source_gen_3():
        for i in range(721):
            yield {"id":i, "text":"dummy3"}
    
    dataloaders = []
    
    for source_gen in [source_gen_1, source_gen_2, source_gen_3]:
        
        dataloaders.append(CachedDataLoader(source_gen, 
                                      cache_chunk_size = 64,
                                      cache_folder = PATH_CACHE))
    
    
    N_SAMPLES = 1000 + 500 + 721
    
    dataloader = CachedDataLoader.merge(*dataloaders)
    
    dataloader.pre_shuffle()
    
    text_samples = []
    
    n_samples = 0
    samples_cumulative_equality = True
    for sample in dataloader:
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"] == n_samples)
        n_samples += 1
        if not n_samples%100:
            text_samples.append(sample["text"])
    
    assert not samples_cumulative_equality # this test ensures order
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert all([text in ["dummy", "dummy2", "dummy3"] for text in text_samples])
    
    
def test_cachedDataloader_from_cache_name():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE)
    
    from_cached_index = dataloader.cache_index_path
    
    del dataloader
    
    dl = CachedDataLoader.from_cached_index(from_cached_index)
    
    dl.pre_shuffle()
    
    n_samples = 0
    samples_cumulative_equality = True
    for sample in dl:
        samples_cumulative_equality = samples_cumulative_equality and (sample["id"] == n_samples)
        n_samples += 1
    
    assert not samples_cumulative_equality # this test ensures order
    assert n_samples == N_SAMPLES
    assert dl.get_n_samples() == N_SAMPLES
    assert sample["text"] == "dummy"


def test_get_bert_embedding_parameters():
    
    get_bert_embeddings = build_bert_embeddings(BERT_CHECKPOINT)
    
    tf.keras.backend.clear_session()
    
def test_get_bert_embedding_parameters_cfg():
    
    cfg = {
        "embeddings":{
            "type":"bert",
            "checkpoint":BERT_CHECKPOINT,
            "bert_layer_index": -1,
        },
    }
    
    get_bert_embeddings = build_bert_embeddings(**cfg)
    
    tf.keras.backend.clear_session()


    
def test_get_bert_embedding_last_layer():

    get_bert_embeddings = build_bert_embeddings(BERT_CHECKPOINT, bert_layer_index=-1)
    
    dummy_data = ["Panda is regarded as Chinese national treasure", 
                  "In 2015, the Nobel Committee for Physiology or Medicine, in its only award for treatments of infectious diseases since six decades prior",
                  "IVM were also observed in two animal models, of SARS-CoV-2 and a related betacoronavirus."]
    
    tokenizer = BertTokenizerFast.from_pretrained(BERT_CHECKPOINT)
    encoding = tokenizer.batch_encode_plus(dummy_data, 
                                          max_length = 64, 
                                          padding="max_length",
                                          truncation=True,
                                          return_token_type_ids = True,
                                          return_attention_mask = True,
                                          return_tensors = "tf")
    
    bert_model = TFBertModel.from_pretrained(BERT_CHECKPOINT,
                                                 output_attentions = False,
                                                 output_hidden_states = True,
                                                 return_dict=True,
                                                 from_pt=True)
    
    control = bert_model(**encoding)["hidden_states"]

    embeddings = get_bert_embeddings(**encoding)
    
    
    assert hasattr(embeddings, "last_hidden_state")
    assert hasattr(embeddings, "pooler_output")
    
    
    assert embeddings["pooler_output"].shape == (3, 768)
    assert embeddings["last_hidden_state"].shape == (3, 64, 768)
    
    assert vector_equals(embeddings["last_hidden_state"], control[-2])
    
    tf.keras.backend.clear_session()
    
    # TODO compare with the bert embeddings by running the normal bert
    
def test_get_bert_embedding_second_to_last_layer():

    get_bert_embeddings = build_bert_embeddings(BERT_CHECKPOINT, bert_layer_index=-2)
    
    dummy_data = ["Panda is regarded as Chinese national treasure", 
                  "In 2015, the Nobel Committee for Physiology or Medicine, in its only award for treatments of infectious diseases since six decades prior",
                  "IVM were also observed in two animal models, of SARS-CoV-2 and a related betacoronavirus."]
    
    tokenizer = BertTokenizerFast.from_pretrained(BERT_CHECKPOINT)
    encoding = tokenizer.batch_encode_plus(dummy_data, 
                                          max_length = 64, 
                                          padding="max_length",
                                          truncation=True,
                                          return_token_type_ids = True,
                                          return_attention_mask = True,
                                          return_tensors = "tf")

    bert_model = TFBertModel.from_pretrained(BERT_CHECKPOINT,
                                                 output_attentions = False,
                                                 output_hidden_states = True,
                                                 return_dict=True,
                                                 from_pt=True)
    
    control = bert_model(**encoding)["hidden_states"]

    embeddings = get_bert_embeddings(**encoding)
    
    
    assert hasattr(embeddings, "last_hidden_state")
    assert hasattr(embeddings, "pooler_output")
    
    
    assert embeddings["pooler_output"].shape == (3, 768)
    assert embeddings["last_hidden_state"].shape == (3, 64, 768)
    
    assert vector_equals(embeddings["last_hidden_state"] , control[-3])
    
    tf.keras.backend.clear_session()
    
    # TODO compare with the bert embeddings by running the normal bert