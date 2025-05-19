from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import asyncio
from functools import partial
from tasks.assessment_api import find_matches
from tasks.most_accurate import find_most_accurate
from tasks.summarization import summarizer
import pandas as pd
app = FastAPI()
templates = Jinja2Templates(directory="templates")



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




@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "job_query": "",
        "result": None,
        "df": None
    })

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, job_query: str = Form(...)):
    job_query = job_query.strip()
    result = None
    df = None

    if job_query:
        summary = summarizer(job_query)
        matches = find_matches(summary)
        matches = find_most_accurate(matches, job_query)

        if matches:
            result = f"Found {len(matches)} related assessment(s) for your input."
            df = pd.DataFrame(matches)
            df = df[["Assessment_Name", "Link", "Duration", "Remote_Testing", "Adaptive_Testing", "Language"]]
            df = df.to_dict(orient="records")
        else:
            result = "No close matches found. Try using a more general or different job description."
    else:
        result = "Please enter a job description to search."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "job_query": job_query,
        "result": result,
        "df": df
    })
    


@app.post("/search", response_model=List[Assessment])
async def search_assessments(data: QueryRequest):
    query =await async_summarizer(data.job_query)
    matches = await async_find_matches(query)
    best_matches = await async_find_most_accurate(matches, query)
    return best_matches
