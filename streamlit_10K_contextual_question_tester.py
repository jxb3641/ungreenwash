import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import glob
import requests
from pathlib import Path
from OpenAIUtils import query_to_summaries, file_to_embeddings, produce_prompt
from CohereUtils import query_to_summaries as query_to_summaries_cohere, file_to_embeddings as file_to_embeddings_cohere, produce_prompt as produce_prompt_cohere
from EDGARFilingUtils import (
    get_all_submission_ids, 
    get_text_from_files_for_submission_id, 
    split_text, 
    filter_chunks, 
    ROOT_DATA_DIR,
    TICKER_TO_COMPANY_NAME,
    QUESTION_TO_CATEGORY
)
import openai
openai.api_key = st.secrets["openai_api_key"]

from transformers import GPT2TokenizerFast
# Used to check the length of the prompt.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

st.set_page_config(layout="wide",page_title="my_title",page_icon="earth")

### Streamlit app starts here
st.title("Play with GPT-3 Completion API and 10-Ks")

list_of_questions = list(QUESTION_TO_CATEGORY.keys())
#Change this question to add to the UI
list_of_questions = [
                     "What are the risks this company faces?",
                     "What does the company do?",
                     "Environmental regulations, environmental laws",
                    ]
relevant_questions = st.multiselect("Select questions to use for search within the text.",
                                    list_of_questions,default=list_of_questions)
temperature = st.number_input("Model Temperature",min_value = 0., max_value = 1., value=0.5, step=0.05)

st.markdown("#")
st.markdown("### GPT-3")
if st.button("Recalc embeddings"):
    st.write("Calculating Embeddings...")
    for fname in ["data/ind_lists/4_food_bev/10k/GIS_0001193125-21-204830_pooled.txt",
                  "data/ind_lists/4_food_bev/10k/PEP_0000077476-22-000010_pooled.txt",
                  "data/ind_lists/11_transportation/10k/F_0000037996-22-000013_pooled.txt",
                  "data/ind_lists/11_transportation/10k/FSR_0001720990-22-000010_pooled.txt"]:
        embeddings = file_to_embeddings(Path(fname),use_cache=False) 
    st.write("Embeddings Saved.")
if st.button("Search for relevant sections to list of questions"):
    st.write(produce_prompt("",query_text=relevant_questions[0]))
    df_completions = query_to_summaries(relevant_questions,temperature,print_responses=False)
    st.table(df_completions)
    df_completions.to_pickle("./risks_responses.pkl")

st.markdown("#")
st.markdown("### Cohere")
if st.button("Recalc embeddings (Cohere)"):
    st.write("Calculating Embeddings...")
    for fname in ["data/ind_lists/4_food_bev/10k/GIS_0001193125-21-204830_pooled.txt"]:
        embeddings = file_to_embeddings_cohere(Path(fname),use_cache=False) 
    st.write("Embeddings Saved.")

if st.button("Search for relevant sections to list of questions (Cohere)"):
    st.write(produce_prompt_cohere("",query_text=relevant_questions[0]))
    df_completions = query_to_summaries_cohere(relevant_questions,temperature,print_responses=False)
    st.table(df_completions)
    df_completions.to_pickle("./risks_responses.pkl")