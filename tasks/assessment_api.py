import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, dotenv_values 

load_dotenv() 

PINECONE_REGION = "us-east-1"
INDEX_NAME = "assessments-index"


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)


embedder =SentenceTransformer('./local_model')


def embed_text(text):
    return embedder.encode(text).tolist()


def find_matches(job_summary, top_k=15):
    query_embedding = embed_text(job_summary)

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = []
    for match in response['matches']:
        metadata = match['metadata']
        matches.append({
            "Assessment Name": metadata.get("Assessment Name", "N/A"),
            "Job Level": metadata.get("job_level", "N/A"),
            "Description": metadata.get("description", "N/A"),
            "Link": metadata.get("link", "N/A"),
            "Duration": metadata.get("duration", "N/A"),
            "Remote Testing": metadata.get("remote_testing", "N/A"),
            "Adaptive Testing": metadata.get("adaptive_testing", "N/A"),
            "Language": metadata.get("language", "N/A"),
            "Score": match.get("score", 0)
        })

    return matches
