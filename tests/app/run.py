import sys
import json
import pandas as pd
import streamlit as st

sys.path.append('/root/textanalyzer')
from src.textanalyzer import MecabTokenization

def token2count(tokens):
    """
    list of Tokens -> counts
    """
    pos = {'KEYPHRASE', 'NNG', 'NNP', 'VV', 'VA', 'XR', 'SL'} # 이건 Init으로 빼자.
    keywords = []
    for tok in tokens:
        if set(tok._pos.split('+')).intersection(pos):
            # stemming
            if '+' in tok._pos:
                for s in tok.expression.split('+'):
                    a, b, _= s.split('/')
                    if b in pos:
                        stem = a
                        p = b
            else:
                stem = tok.text
                p = tok._pos

            # lemmantization
            if p in {'VV', 'VA'}:
                stem = f"{stem}다"
            keywords.append(f"{stem} {p}") 
            
    return ' -- '.join(keywords)

if __name__ == "__main__":
    
    tokenizer = MecabTokenization(custom_dir='/root/custom_dict')
    st.title("토크나이저 검수")
    text = st.text_area("텍스트 입력:")
    tokens = tokenizer(text).Tokens
    result = token2count(tokens)
#     result = ' -- '.join([tok.text for tok in tokens])
    st.markdown("## 토크나이즈 결과")
    st.markdown("> ` -- `는 토크나이즈 결과의 구분자. <br>")
    st.write(result)
