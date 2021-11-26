Module polus.ner.elements
=========================

Functions
---------

    
`add_offset_to_spans(spans, offset)`
:   

    
`assert_valid_span(span)`
:   

    
`get_entities_within_span(entities, span)`
:   

    
`get_non_overlapping_and_overlapping_entities(entities)`
:   

    
`merge_collections(*args, make_deepcopy=False)`
:   

    
`sort_identifiers(identifiers)`
:   

    
`span_contains_span(span1, span2)`
:   

    
`span_overlaps_span(span1, span2)`
:   

Classes
-------

`Collection()`
:   A set of documents, using the above defined Document class.
    Each document is associated with a specific identifier.
    The identifier has to be a string.
    For example, a PMCID (PubMed Central ID) is a valid identifier.

    ### Methods

    `add(self, i, d, make_deepcopy=False)`
    :   Add document with the respective identifier.

    `add_metadata(self, corpus, group)`
    :   Allow the collection to know the corpus where it belongs.

    `clear_entities(self)`
    :   Remove the entities from all the documents. This is useful in
        the inference phase. It serves as a sanity measure.

    `get(self, i)`
    :

    `ids(self)`
    :   Return a sorted list with the identifiers.

    `json(self)`
    :

    `pretty_json(self)`
    :

`Document(identifier)`
:   A Document contains an identifier and a PassageOrderedList object.

    ### Methods

    `add_passage(self, p)`
    :

    `add_passages(self, passages)`
    :

    `clear_entities(self)`
    :   Remove all the entities from the document.

    `entities(self, sort=False, make_deepcopy=False)`
    :

    `get_entity_set(self)`
    :

    `iis(self)`
    :

    `indexing_identifiers(self, sort=False, make_deepcopy=False)`
    :

    `json(self)`
    :

    `nes(self)`
    :

    `passages_entities(self, sort=False, make_deepcopy=False)`
    :

    `passages_spans(self)`
    :

    `passages_texts(self)`
    :

    `pretty_json(self)`
    :

    `set_entities(self, entities)`
    :

    `set_indexing_identifiers(self, indexing_identifiers)`
    :

    `set_mesh_indexing_identifiers(self, mesh_indexing_identifiers)`
    :

    `span(self)`
    :

    `text(self)`
    :

`Entity(text, span, typ)`
:   An Entity has a textual mention, a span (start and end offsets),
    and a type.
    
    Example (taken from NLM-Chem, dev subset, PMCID 4200806):
        text: 'tyrosine'
        span: (2914, 2922)
        typ: 'Chemical'
    
    >>> Entity('tyrosine', (2914, 2922), 'Chemical')
    
    Example (taken from NLM-Chem, train subset, PMCID 5600090):
        text: 'MIDA boronate ester'
        span: (758, 777)
        typ: 'Chemical'
    
    >>> Entity('MIDA boronate ester', (758, 777), 'Chemical')

    ### Descendants

    * polus.ner.elements.NormalizedEntity

    ### Methods

    `json(self, i=1)`
    :

    `pretty_json(self, i=1)`
    :

    `to_normalized_entity(self)`
    :

`EntitySet(entities=None)`
:   A set of Entity objects.
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
    
    >>> e1 = Entity('Carbaryl', (43, 51), 'Chemical')
    >>> e2 = Entity('Naphthalene', (52, 63), 'Chemical')
    >>> e3 = Entity('Chlorpyrifos', (68, 80), 'Chemical')
    >>>
    >>> es = EntitySet([e1, e2, e3])

    ### Descendants

    * polus.ner.elements.NormalizedEntitySet

    ### Methods

    `add(self, e)`
    :

    `difference(self, other, make_deepcopy=False)`
    :

    `get(self, sort=False, make_deepcopy=False)`
    :

    `has(self, e)`
    :

    `intersection(self, other, make_deepcopy=False)`
    :

    `json(self, start=1)`
    :

    `pretty_json(self, start=1)`
    :

    `to_normalized_entity_set(self)`
    :

    `union(self, other, make_deepcopy=False)`
    :

    `update(self, entities)`
    :

`IndexingIdentifier(identifier, typ)`
:   An IndexingIdentifier has a single identifier (for example,
    a MeSH ID), and a type (for example, "MeSH_Indexing_Chemical").
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
      identifier: 'MESH:D009281'
      typ: 'MeSH_Indexing_Chemical'
    
    >>> IndexingIdentifier('MESH:D009281', 'MeSH_Indexing_Chemical')

    ### Methods

    `json(self, i=1)`
    :

    `pretty_json(self, i=1)`
    :

`IndexingIdentifierSet(indexing_identifiers=None)`
:   A set of IndexingIdentifier objects.
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
    
    >>> ii1 = IndexingIdentifier('MESH:D009281', 'MeSH_Indexing_Chemical')
    >>> ii2 = IndexingIdentifier('MESH:D009284', 'MeSH_Indexing_Chemical')
    >>> ii3 = IndexingIdentifier('MESH:D011728', 'MeSH_Indexing_Chemical')
    >>> ii4 = IndexingIdentifier('MESH:C031721', 'MeSH_Indexing_Chemical')
    >>>
    >>> iis = IndexingIdentifierSet([ii1, ii2, ii3, ii4])

    ### Methods

    `add(self, ii)`
    :

    `difference(self, other, make_deepcopy=False)`
    :

    `get(self, sort=False, make_deepcopy=False)`
    :

    `has(self, ii)`
    :

    `intersection(self, other, make_deepcopy=False)`
    :

    `json(self, start=1)`
    :

    `pretty_json(self, start=1)`
    :

    `union(self, other, make_deepcopy=False)`
    :

    `update(self, indexing_identifiers)`
    :

`NormalizedEntity(text, span, typ, identifiers=None)`
:   A NormalizedEntity inherits from Entity.
    
    Besides having a textual mention, a span, and a type, additionally
    has a list of identifiers (for normalization).
    
    Note: it has to be a list of identifiers (not a set) because the
          order of the identifiers matters. For example, if the entity
          text mention refers to three diferent terms, its identifiers
          follow their order of appearance (see Example 2 below).
    
    Example 1 (taken from NLM-Chem, dev subset, PMCID 4200806):
        text: 'tyrosine'
        span: (2914, 2922)
        typ: 'Chemical'
        identifiers: ['MESH:D014443']
    
    >>> NormalizedEntity('tyrosine', (2914, 2922), 'Chemical', ['MESH:D014443'])
    
    Example 2 (taken from NLM-Chem, train subset, PMCID 5600090):
        text: 'MIDA boronate ester'
        span: (758, 777)
        typ: 'Chemical'
        identifiers: ['MESH:C533766', 'MESH:D001897', 'MESH:D004952']
    
    >>> NormalizedEntity('MIDA boronate ester', (758, 777), 'Chemical', ['MESH:C533766', 'MESH:D001897', 'MESH:D004952'])
    
    Example 3 (taken from NLM-Chem, train subset, PMCID 4988499)
        text: 'cyclic, aromatic, and monoterpenoid enones, enals, and enols'
        span: (2793, 2853)
        typ: 'Chemical'
        identifiers: ['MESH:D007659', 'MESH:D000447', '-']
    
    >>> NormalizedEntity('cyclic, aromatic, and monoterpenoid enones, enals, and enols', (2793, 2853), 'Chemical', ['MESH:D007659', 'MESH:D000447', '-'])

    ### Ancestors (in MRO)

    * polus.ner.elements.Entity

    ### Methods

    `json(self, i=1)`
    :

    `set_identifiers(self, identifiers)`
    :

    `to_entity(self)`
    :

`NormalizedEntitySet(entities=None)`
:   A set of NormalizedEntity objects.
    
    Example (taken from NLM-Chem, train subset, PMCID 1253656):
    
    >>> ne1 = NormalizedEntity('Carbaryl', (43, 51), 'Chemical', ['MESH:D012721'])
    >>> ne2 = NormalizedEntity('Naphthalene', (52, 63), 'Chemical', ['MESH:C031721'])
    >>> ne3 = NormalizedEntity('Chlorpyrifos', (68, 80), 'Chemical', ['MESH:D004390'])
    >>>
    >>> nes = NormalizedEntitySet([ne1, ne2, ne3])

    ### Ancestors (in MRO)

    * polus.ner.elements.EntitySet

    ### Methods

    `add(self, e)`
    :

    `difference(self, other, make_deepcopy=False)`
    :

    `has(self, e)`
    :

    `intersection(self, other, make_deepcopy=False)`
    :

    `to_entity_set(self)`
    :

    `union(self, other, make_deepcopy=False)`
    :

`Passage(text, span, typ, section_type)`
:   A Passage is initialized with a text, a span (start and end
    offsets), a type (abstract, fig_caption, footnote, front,
    paragraph, ref, table_caption, title, etc), and a section type
    (ABSTRACT, INTRO, METHODS, RESULTS, etc). Note that, in the
    NLM-Chem dataset, frequently the section types are undefined.
    
    At first, a Passage has no annotations. But these can be added
    iteratively. Annotations are NormalizedEntity or IndexingIdentifier
    objects.

    ### Methods

    `add_entities(self, entities)`
    :

    `add_entity(self, e)`
    :

    `add_indexing_identifier(self, ii)`
    :

    `add_indexing_identifiers(self, indexing_identifiers)`
    :

    `entities(self, sort=False, make_deepcopy=False)`
    :

    `get_entity_set(self)`
    :

    `indexing_identifiers(self, sort=False, make_deepcopy=False)`
    :

    `json(self, nes_i=1, iis_i=1)`
    :

    `pretty_json(self, nes_i=1, iis_i=1)`
    :

`PassageOrderedList()`
:   This class contains a list of Passage objects ordered by their
    span offsets. Also Passage objects cannot overlap, that is, the
    Passage objects must be disjoint.
    
    Example (made-up):
    
    >>> p1 = Passage('The title.', (0, 10), 'front', 'TITLE')
    >>> p2 = Passage('An abstract.', (11, 23), 'abstract', 'ABSTRACT')
    >>> p3 = Passage('A first paragraph.', (24, 42), 'paragraph', 'INTRO')
    >>> p4 = Passage('A second paragraph.', (43, 62), 'paragraph', 'INTRO')
    >>>
    >>> pol = PassageOrderedList()
    >>> pol.add(p3)
    >>> pol.add(p2)
    >>> pol.add(p4)
    >>> pol.add(p1)

    ### Methods

    `add(self, p)`
    :

    `entities(self, sort=False, make_deepcopy=False)`
    :

    `get_entity_set(self)`
    :

    `iis(self)`
    :

    `indexing_identifiers(self, sort=False, make_deepcopy=False)`
    :

    `json(self)`
    :

    `nes(self)`
    :

    `passages_entities(self, sort=False, make_deepcopy=False)`
    :

    `passages_spans(self)`
    :

    `passages_texts(self)`
    :

    `pretty_json(self)`
    :

    `span(self)`
    :   Return the span containing all the passages.
        The start offset is always zero.

    `text(self)`
    :   Return the whole text containing all the text passages.