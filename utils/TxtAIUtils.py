import re
from utils.EDGARFilingUtils import split_text, filter_chunks
from txtai.embeddings import Embeddings
import os
import re
from pathlib import Path

SECTION_DELIM_PATTERN = re.compile("####.+") # for pooled 10k files
basepath = "10ks"

huggingface_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# content == true returns the text as well
embeddings = Embeddings({"path": huggingface_embedding_model, "content": True, "objects": True})


def get_text_chunks(filepath):
    raw_text = filepath.read_text(encoding="utf-8").replace("$","\$")
    if "pooled" in str(filepath): # pooled 10-K files are split into item1, item1a, item7 using a delimiter. 
        items = re.split(SECTION_DELIM_PATTERN,raw_text)
        text_chunks = []
        for item in items:
            section_chunked = split_text(item,form_type="10KItemsOnly")
            for chunk in section_chunked:
                text_chunks.append(chunk)
    else:
        text_chunks = filter_chunks(split_text(raw_text))
    return text_chunks

def get_company_chunks(company):
    company_chunks = []
    for filename in os.listdir(os.path.join(basepath,company)):
        filepath = Path(os.path.join(basepath,company, filename))
        text_chunks = get_text_chunks(filepath)
        for chunk in text_chunks:
            company_chunks.append({"filename": filename, "chunk": chunk})
    return company_chunks

def store_embeddings():
    for company in os.listdir(basepath):
        output_path = os.path.join("embeddings", company)
        if os.path.exists(output_path):
            continue

        company_chunks_full = get_company_chunks(company)
        company_chunks = [val["chunk"] for val in company_chunks_full]

        embeddings.index([(i, chunk, None) for i, chunk in enumerate(company_chunks)])
        # We can save to a cloud storage platform here instead of locally
        embeddings.save(output_path)

def get_file_from_index(index, company):
    company_chunks = get_company_chunks(company)
    return company_chunks[int(index)]["filename"]

def load_embeddings(company):
    embeddings.load(f"embeddings/{company}")

def semantic_search(query, company):
    load_embeddings(company)
    result = embeddings.search(query, 1)[0]
    return result