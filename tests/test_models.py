from polus.models import split_bert_model
from transformers import TFBertModel, AutoTokenizer
import tensorflow as tf
from tests.utils import vector_equals

def test_validate_output_of_split_bert():
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    bert_model = TFBertModel.from_pretrained("bert-base-uncased",
                                                 output_attentions = False,
                                                 output_hidden_states = True,
                                                 return_dict=True,
                                                 from_pt=True)
    
    sample = "hello, this is a sample that i want to tokenize"

    inputs = tokenizer.encode_plus(sample,
                                       padding = "max_length",
                                       truncation = True,
                                       max_length = 50,
                                       return_attention_mask = True,
                                       return_token_type_ids = True,
                                       return_tensors = "tf",
                                      )

    control = bert_model(**inputs)["hidden_states"]
    
    pre_model, post_model = split_bert_model(bert_model, -2, init_models=False)
    
    hidden_states = pre_model(**inputs)["last_hidden_state"]
    out = post_model(hidden_states=hidden_states, attention_mask=inputs["attention_mask"])["last_hidden_state"]
    
    assert vector_equals(hidden_states, control[-3])
    assert vector_equals(out, control[-1]) 
    
    

    


