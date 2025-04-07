from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from assessement_retriever import find_matches
from gen_model import run
from most_accurate import  find_most_accurate  # Your final LLM filter logic

app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str

# Response model
class Assessment(BaseModel):
    Assessment_Name: str
    Link: str
    Duration: str
    Remote_Testing: str
    Adaptive_Testing: str
    Language: str

@app.post("/search", response_model=List[Assessment])
def search_assessments(data: QueryRequest):
    query = data.query
    matches = find_matches(query)  # embedding-based shortlist
    best_matches = find_most_accurate(matches, query)  # LLM-ranked top N

    return best_matches
