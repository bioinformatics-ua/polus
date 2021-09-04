from core import BaseLogger
from copy import deepcopy
from polus.ner.bio import decode_bio
from polus.ner.bio import get_bio
from polus.ner.elements import get_entities_within_span

class SequenceDecoder(BaseLogger):
    
    def __init__(self, corpora):
        
        super().__init__()
        self.corpora = {str(corpus): corpus for corpus in corpora}
        
        # auxiliar var that holds a dictionary that can be built in a batch-wise fashion
        self.documents_dict = {}
        
        #
        # Load into memory the gold standard (true) texts and entity
        # sets.
        #
        self.documents = dict()
        for corpus_str, corpus in self.corpora.items():
            self.documents[corpus_str] = dict()
            for group, collection in corpus:
                self.documents[corpus_str][group] = dict()
                for i, d in collection:
                    self.documents[corpus_str][group][i] = dict()
                    self.documents[corpus_str][group][i]['text'] = d.text()
                    # self.documents[corpus_str][group][i]['nes'] = d.nes()
                    self.documents[corpus_str][group][i]['es'] = d.get_entity_set()
    
    def clear_state(self):
        self.documents_dict = {}
    
    def samples_from_batch(self, samples):
         #
        # Unbatching in Python.
        #
        _samples = []
        
        if isinstance(samples, dict):
            samples = [samples]
            
        for sample in samples:
            key = list(sample.keys())[0]
            batch_size = sample[key].shape[0]
            for i in range(batch_size):
                _samples.append({k: v[i] for k, v in sample.items()})
        samples = _samples
        
        for data in samples:
            corpus = data['corpus'].numpy().decode()
            group = data['group'].numpy().decode()
            identifier = data['identifier'].numpy().decode()
            spans = data['spans'].numpy().tolist()
            #
            # Convert predicted integer values tags to predicted tags.
            #
            tags_pred = [INT2TAG[i] for i in data['tags_int_pred'].numpy().tolist()]
            
            is_prediction = data['is_prediction'].numpy().tolist()
            
            if corpus not in self.documents_dict:
                self.documents_dict[corpus] = dict()
            if group not in self.documents_dict[corpus]:
                self.documents_dict[corpus][group] = dict()
            if identifier not in self.documents_dict[corpus][group]:
                self.documents_dict[corpus][group][identifier] = {
                    'spans': list(), 'tags': list()}
            
            #
            # Select only the values that were marked for prediction.
            #
            filtered_spans = list()
            filtered_tags = list()
            for s, t, p in zip(spans, tags_pred, is_prediction):
                if p == 1:
                    filtered_spans.append(s)
                    filtered_tags.append(t)
            
            self.documents_dict[corpus][group][identifier]['spans'].extend(filtered_spans)
            self.documents_dict[corpus][group][identifier]['tags'].extend(filtered_tags)
            
    def decode(self):

        counts = {
            'tags': 0,
            'inside_tag_after_other_tag': 0,
            'inside_tag_with_different_entity_type': 0,
        }
        
        #
        # The spans and the respective tags are assumed to be already
        # ordered.
        #
        for corpus in self.documents_dict:
            for group in self.documents_dict[corpus]:
                for identifier in self.documents_dict[corpus][group]:
                    #
                    # Get the original text from the gold standard (true)
                    # document.
                    #
                    text = self.documents[corpus][group][identifier]['text']
                    
                    spans = self.documents_dict[corpus][group][identifier]['spans']
                    tags = self.documents_dict[corpus][group][identifier]['tags']
                    #
                    # Given the predicted tags, the respective spans,
                    # and the original (true) text get the predicted
                    # entity set.
                    #
                    es, c = decode_bio(tags, spans, text, allow_errors=True)
                    self.documents_dict[corpus][group][identifier]['es'] = es
                    
                    counts['tags'] += c['tags']
                    counts['inside_tag_after_other_tag'] += c['inside_tag_after_other_tag']
                    counts['inside_tag_with_different_entity_type'] += c['inside_tag_with_different_entity_type']
        
        s = 'Statistics about the BIO decoding process: tags={}, inside_tag_after_other_tag={}, inside_tag_with_different_entity_type={}.'.format(
            counts['tags'], counts['inside_tag_after_other_tag'], counts['inside_tag_with_different_entity_type'])
        
        self.logger.info(s)
    
    def decode_from_samples(self, samples):
        #
        # To keep track of the number of BIO decoding errors.
        #
        
        # here we assume a gigant batch that contains all the dataset
        self.samples_from_batch(samples)
        
        self.decode()
            
    def evaluate_ner_from_sample(self, samples):
        
        self.decode_from_samples(samples)
        
        return self._evaluate_ner()
    
    def evaluate_ner(self):
        
        self.decode()
        
        return self._evaluate_ner()
    
    def _evaluate_ner(self):
        
        true_list = list()
        pred_list = list()
        
        for corpus in self.documents_dict:
            for group in self.documents_dict[corpus]:
                for identifier in self.documents_dict[corpus][group]:
                    
                    true_es = self.documents[corpus][group][identifier]['es']
                    pred_es = self.documents_dict[corpus][group][identifier]['es']
                    
                    true_list.append(true_es)
                    pred_list.append(pred_es)
        
        results = eval_list_of_entity_sets(true_list, pred_list)
        
        # clear the document state
        self.clear_state()
        
        return results
    
    def _get_collections(self):
                
        collections = dict()
        for corpus in self.documents_dict:
            collections[corpus] = dict()
            for group in self.documents_dict[corpus]:
                #
                # This is important: make a deepcopy to not modify the
                #                    original corpora.
                #
                collections[corpus][group] = deepcopy(self.corpora[corpus][group])
                #
                # Sanity measure: remove all the entities from the
                #                 collection.
                #
                collections[corpus][group].clear_entities()
        
        for corpus in self.documents_dict:
            for group in self.documents_dict[corpus]:
                for identifier in self.documents_dict[corpus][group]:
                    nes = self.documents_dict[corpus][group][identifier]['es'].to_normalized_entity_set()
                    entities = nes.get()
                    collections[corpus][group][identifier].set_entities(entities)
        
        # clear the document state
        self.clear_state()
        
        return collections
    
    def get_collections_from_samples(self, samples):
        r"""
        Return a dictionary with Collection objects.
        The first-level key is the corpus name.
        The second-level key is the group name.
        
        Each collection contains the predicted entities derived from
        the predicted samples (that have the predicted BIO tags).
        """
        self.decode_from_samples(samples)
        
        return self._get_collections()
    
    def get_collections(self):
        r"""
        Return a dictionary with Collection objects.
        The first-level key is the corpus name.
        The second-level key is the group name.
        
        Each collection contains the predicted entities derived from
        the predicted samples (that have the predicted BIO tags).
        """
        self.decode()
        
        return self._get_collections()