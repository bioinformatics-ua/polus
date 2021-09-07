from polus.experimental.data import DataLoader, CachedDataLoader, build_bert_embeddings
import pytest
import glob
import os
import shutil

from transformers import BertTokenizerFast
import tensorflow as tf

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

def test_dataloader_nocache():
    
    
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = DataLoader(source_gen)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
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
                            accelerated_map_f=mapping_f)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2

    
def test_cachedDataloader():
    
    N_SAMPLES = 1000
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    
    # assert cache files
    cache_files = os.listdir(PATH_CACHE)
    assert f"{dataloader.cache_base_name}.index" in cache_files
    
    for data_files in dataloader.cache_index["files"]:
        assert os.path.basename(data_files) in cache_files
    
    
def test_cachedDataloader_mapping_f_cache():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE,
                                  accelerated_map_f=mapping_f)
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["id"] == N_SAMPLES-1
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    
    
def test_cachedDataloader_mapping_f_cache_pre_shuffle():
    
    N_SAMPLES = 1000
    
    def mapping_f(sample):
        sample["new_entry"] = sample["id"]*2
        return sample

    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE,
                                  accelerated_map_f=mapping_f)
    
    dataloader.pre_shuffle()
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    
def test_cachedDataloader_mapping_f_and_py_sample_f_cache_pre_shuffle():
    
    N_SAMPLES = 1000
    
    def mapping_f(samples):
        samples["new_entry"] = samples["id"]*2
        return samples
    
    def samples_map(sample):
        sample["new_entry_sample_map"] = str(int(sample["id"].numpy()))
        return sample
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE,
                                  accelerated_map_f=mapping_f,
                                  py_sample_map_f=samples_map)
    
    dataloader.pre_shuffle()
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    assert sample["new_entry_sample_map"] == str(int(sample["id"].numpy()))
    
def test_cachedDataloader_mapping_f_and_py_sample_f_tf_sample_f_cache_pre_shuffle():
    
    N_SAMPLES = 1000
    
    def mapping_f(samples):
        samples["new_entry"] = samples["id"]*2
        return samples
    
    def tf_samples_map(sample):
        sample["tf_new_entry"] = sample["id"] + 1 
        return sample
    
    def samples_map(sample):
        sample["new_entry_sample_map"] = str(int(sample["id"].numpy()))
        return sample
    
    def source_gen():
        for i in range(N_SAMPLES):
            yield {"id":i, "text":"dummy"}
            
    dataloader = CachedDataLoader(source_gen, 
                                  cache_chunk_size = 256,
                                  cache_folder = PATH_CACHE,
                                  accelerated_map_f=mapping_f,
                                  py_sample_map_f=samples_map,
                                  tf_sample_map_f=tf_samples_map)
    
    dataloader.pre_shuffle()
    
    n_samples = 0
    for sample in dataloader:
        n_samples += 1
    
    assert n_samples == N_SAMPLES
    assert dataloader.get_n_samples() == N_SAMPLES
    assert sample["text"] == "dummy"
    assert sample["new_entry"] == sample["id"]*2
    assert sample["new_entry_sample_map"] == str(int(sample["id"].numpy()))
    assert sample["tf_new_entry"] == sample["id"] + 1
    

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

    get_bert_embeddings = build_bert_embeddings(BERT_CHECKPOINT)
    
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

    embeddings = get_bert_embeddings(input_ids=encoding["input_ids"], 
                                     token_type_ids=encoding["token_type_ids"],
                                     attention_mask=encoding["attention_mask"])
    
    
    assert hasattr(embeddings, "last_hidden_state")
    assert hasattr(embeddings, "pooler_output")
    
    
    assert embeddings["pooler_output"].shape == (3, 768)
    assert embeddings["last_hidden_state"].shape == (3, 64, 768)
    
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

    embeddings = get_bert_embeddings(input_ids=encoding["input_ids"], 
                                     token_type_ids=encoding["token_type_ids"],
                                     attention_mask=encoding["attention_mask"])
    
    
    assert hasattr(embeddings, "last_hidden_state")
    assert hasattr(embeddings, "pooler_output")
    
    
    assert embeddings["pooler_output"].shape == (3, 768)
    assert embeddings["last_hidden_state"].shape == (3, 64, 768)
    
    tf.keras.backend.clear_session()
    
    # TODO compare with the bert embeddings by running the normal bert