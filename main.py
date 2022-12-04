from fastapi import FastAPI
from utils.TxtAIUtils import store_embeddings, semantic_search, get_file_from_index
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store_embeddings()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/companies/")
def get_companies():
    return ["Ford", "Fisker", "General Mills", "Pepsi"]

@app.post("/api/batch/{company}/")
def ask_questions(company: str, questions: List[str]):
    return [get_question_answer(company, question) for question in questions]

@app.post("/api/{company}/")
def ask_questions(company: str, question: str):
    return get_question_answer(company, question)

def get_question_answer(company, question):
    ret = semantic_search(question, company)
    ret["filename"] = get_file_from_index(ret["id"], company)
    return ret

