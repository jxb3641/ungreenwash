from txtai.embeddings import Embeddings
import streamlit as st
import os
import re
from pathlib import Path
from TxtAIUtils import store_embeddings, semantic_search


# embeddings.index([(0, "Correct", None), (1, "Not what we hoped", None)])
# ret = embeddings.search("positive", 1)[0]

st.set_page_config(layout="wide",page_title="my_title",page_icon="earth")

# st.write(ret)
store_embeddings()

company = "Ford"
semantic_search_query = "risk of climate"

ret = semantic_search(semantic_search_query, company)
st.write(ret)




        
        