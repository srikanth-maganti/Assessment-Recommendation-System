from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import asyncio
from functools import partial
from assessment_api import find_matches
from most_accurate import find_most_accurate
from summarization import summarizer

app = FastAPI()




class QueryRequest(BaseModel):
    job_query: str


class Assessment(BaseModel):
    Assessment_Name: str
    Link: str
    Duration: str
    Remote_Testing: str
    Adaptive_Testing: str
    Language: str




async def async_find_matches(job_query: str, top_k: int = 15):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(find_matches, job_query, top_k))


async def async_find_most_accurate(matches, job_query: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(find_most_accurate, matches, job_query))


async def async_summarizer(job_query: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(summarizer, job_query))

@app.get("/")
def root():
    return {"message": "Welcome to SHL Assessment Recommendation Engine!"}


@app.post("/debug")
async def debug_endpoint(data: QueryRequest):
    matches = await async_find_matches(data.job_query)
    return {"matches": matches[:3]}


@app.post("/search", response_model=List[Assessment])
async def search_assessments(data: QueryRequest):
    query =await async_summarizer(data.job_query)
    matches = await async_find_matches(query)
    best_matches = await async_find_most_accurate(matches, query)
    return best_matches
