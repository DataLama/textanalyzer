complex_korean_keyword_patterns = [
    # '(XR|SL)'
    [{"TAG":{"IN":["XR", "SL"]}}],
    
    # '(NNG|NNP)*' 
    [{"TAG":{"IN":["NNG", "NNP"]}, "SPACY": False, "OP":"{0,5}"}, 
        {"TAG":{"IN":["NNG", "NNP"]}}],
    
    # 'SL+(NNG|NNP)*'
    [{"TAG": "SL"}, {"TAG":{"IN":["NNG", "NNP"]}, "SPACY": False, "OP":"{1,5}"}, 
        {"TAG":{"IN":["NNG", "NNP"]}}],
    
    # '(NR|SN|MM)+NNBC',
    [{"TAG": "NR"}],
    
    [{"TAG": {"IN":["NR", "SN", "MM"]}, "SPACY" : False},{"TAG": "NNBC"}],
    
    [{"TAG": {"IN":["NR", "SN"]}, "SPACY" : False}, 
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TAG": "NNBC"}],

    [{"TAG": {"IN":["NR", "SN"]}, "SPACY" : False}, 
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TAG": "NNBC"}],

    [{"TAG": {"IN":["NR", "SN"]}, "SPACY" : False}, 
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TAG": "NNBC"}],
    
    [{"TAG": {"IN":["NR", "SN"]}, "SPACY" : False}, 
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TEXT": ",", "SPACY" : False}, {"TEXT": {"REGEX":"^[0-9][0-9][0-9]$"}, "SPACY" : False},
        {"TAG": "NNBC"}],

    # '(NR|SN)+NNBC+XSN',
    [{"TAG": {"IN":["NR", "SN"]}, "SPACY" : False},{"TAG": "NNBC", "SPACY" : False}, {"TAG": "XSN"}],
    
    # '(NNG|NNP|XR)*+(XSN|NNB)'
    [{"TAG":{"IN":["NNG", "NNP", "XR"]}, "SPACY": False, "OP":"{1,5}"}, 
        {"TAG":{"IN":["XSN", "NNB"]}}],

    # 'XPN+(NNG|NNP|XR)*'
    [{"TAG": "XPN"}, {"TAG":{"IN":["NNG", "NNP", "XR"]}, "SPACY": False, "OP":"{1,5}"}, 
        {"TAG":{"IN":["NNG", "NNP", "XR"]}}],

    # 'XPN+(NNG|NNP|XR)*+XSN'
    [{"TAG": "XPN"}, {"TAG":{"IN":["NNG", "NNP", "XR"]}, "SPACY": False, "OP":"{1,5}"}, 
        {"TAG":"XSN"}],

    # 'MM+NNG'
    [{"TAG": "MM", "SPACY" : False},{"TAG": "NNG"}],

    # VERB Keyword Pattern
    [{"TAG":{"IN":["VV", "VA"]}}]
]