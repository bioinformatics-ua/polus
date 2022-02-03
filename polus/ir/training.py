import tensorflow as tf
from polus.training import BaseTrainer
from polus.core import get_jit_compile

class EfficientDenseRetrievalTrainer(BaseTrainer):
    """
    This kind of trainer will conduct the negative sample based on the positive examples already presented
    in the batch.
    
    For instance:
    
    sample-1 -> (q_1, d_pos_1)
    sample-2 -> (q_2, d_pos_2)
    ...
    sample-N -> (q_N, d_pos_N)
    
    If we consider sample-2 as the example the d_pos_2 correspond to its positive doc, while d_pos_1;d_pos_3;...;doc_pos_N 
    can be used as negatives samples (It is important to check if the data as no conflit)
    """
    
    def __init__(self, 
                 model,
                 compute_scores,
                 k_negatives = 0,
                 trainable_weights = None,
                 *args,
                 **kwargs):
        
        # create a wrapper model
        self.compute_scores = compute_scores
        self.k_negatives = k_negatives
        
        if trainable_weights is None:
            # concatenation of all the model trainable variables 
            self.trainable_weights = model.trainable_weights
        else:
            self.trainable_weights = trainable_weights
        
        super().__init__(model, *args, **kwargs)
    
    def __str__(self):
        return 'SimilarityTrainer'
    
    
    @tf.function(input_signature=[{"input_ids": tf.TensorSpec([None, None], dtype=tf.int32), "attention_mask":tf.TensorSpec([None,None], dtype=tf.int32)}, 
                                   {"input_ids": tf.TensorSpec([None, None], dtype=tf.int32), "attention_mask":tf.TensorSpec([None,None], dtype=tf.int32)}])
    def _forward_base_model(self, question, positive_doc):
        self.logger.debug("_forward_base_model function was traced")
        
        query_representation = self.model.encode_query(question, training=True) # B, E
        positive_doc_representation = self.model.encode_document(positive_doc, training=True) # B, E or B, L, E
        
        return query_representation, positive_doc_representation#self.compute_scores(query_representation, positive_doc_representation)


    @tf.function(input_signature=[{"input_ids": tf.TensorSpec([None, None], dtype=tf.int32), "attention_mask":tf.TensorSpec([None,None], dtype=tf.int32)}, 
                                   {"input_ids": tf.TensorSpec([None, None], dtype=tf.int32), "attention_mask":tf.TensorSpec([None,None], dtype=tf.int32)},
                                   {"input_ids": tf.TensorSpec([None, None, None], dtype=tf.int32), "attention_mask":tf.TensorSpec([None, None, None], dtype=tf.int32)}])
    def _forward_base_model_w_negatives(self, question, positive_doc, negative_docs):
        self.logger.debug("_forward_base_model_w_negatives function was traced")

        query_representation = self.model.encode_query(question, training=True) # B, E
        positive_doc_representation = self.model.encode_document(positive_doc, training=True) # B, E or B, L, E
        negative_docs_representation = [ tf.expand_dims(self.model.encode_document({"input_ids":negative_docs["input_ids"][:,i,:], "attention_mask":negative_docs["attention_mask"][:,i,:]}, training=True), axis=0) 
                                        for i in range(self.k_negatives)]
            
        return query_representation, positive_doc_representation, tf.concat(negative_docs_representation, axis=0)#self.compute_scores(query_representation, positive_doc_representation, *negative_docs_representation)
    
    def forward_without_grads(self, question, positive_doc, negative_doc = None):
        
        if negative_doc is None:
            return self._forward_base_model(question, positive_doc)
        else:
            self.k_negatives = negative_doc["input_ids"].shape[1]
            return self._forward_base_model_w_negatives(question, positive_doc, negative_doc)
    
    @tf.function(input_signature=[ tf.TensorSpec([None, None], dtype=tf.float32), 
                                   tf.TensorSpec([None, None], dtype=tf.float32)])
    def _forward_trainable_model(self, question_rep, positive_doc_rep):
        self.logger.debug("_forward_loss_pass function was traced")

        query_representation = self.model.query_projection(question_rep, training=True) # B, E
        positive_doc_representation = self.model.document_projection(positive_doc_rep, training=True)
        
        if self.post_process_logits is not None:
            query_representation = self.post_process_logits(query_representation)
            positive_doc_representation = self.post_process_logits(positive_doc_representation)
        
        return self.compute_scores(query_representation, positive_doc_representation)

    @tf.function(input_signature=[ tf.TensorSpec([None, None], dtype=tf.float32), 
                                   tf.TensorSpec([None, None], dtype=tf.float32),
                                   tf.TensorSpec([None, None, None], dtype=tf.float32)])
    def _forward_trainable_model_w_negatives(self, question_rep, positive_doc_rep, negative_docs_rep):
        self.logger.debug("_forward_loss_pass_w_negatives function was traced")

        query_representation = self.model.query_projection(question_rep, training=True) # B, E
        positive_doc_representation = self.model.document_projection(positive_doc_rep, training=True) # B, E or B, L, E
        negative_docs_representation = [ self.model.document_projection(negative_docs_rep[i,:], training=True) 
                                        for i in range(self.k_negatives)]
        
        if self.post_process_logits is not None:
            query_representation = self.post_process_logits(query_representation)
            positive_doc_representation = self.post_process_logits(positive_doc_representation)
            negative_docs_representation = [ self.post_process_logits(negative_representation) for negative_representation in negative_docs_representation]
            
        return self.compute_scores(query_representation, positive_doc_representation, *negative_docs_representation)
        
    def forward_with_grads(self, question, positive_doc, negative_doc = None):

        ## Under normal circunstancies only one branch would be executed, therefor only one computational graph
        if negative_doc is None:
            pos_scores, neg_scores = self._forward_trainable_model(question, positive_doc)
        else:
            pos_scores, neg_scores = self._forward_trainable_model_w_negatives(question, positive_doc, negative_doc)

        return pos_scores, neg_scores#self.loss(pos_scores, neg_scores)

