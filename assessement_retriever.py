import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Limit to 1 worker thread to avoid Render CPU/thread limits
settings = Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
)

db_client = chromadb.PersistentClient(settings)
# db_client = chromadb.PersistentClient(path="./chroma_db")


collection = db_client.get_or_create_collection(name="assessments")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to embed text
def embed_text(text):
    return embedder.encode(text).tolist()

# Function to find top-k similar assessment matches
def find_matches(job_summary, top_k=20):
    query_embedding = embed_text(job_summary)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract matching documents and metadata
    matches = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        match = {
            "Assessment Name": doc,
            "Job Level": metadata.get("job_level", "N/A"),
            "Description": metadata.get("description", "N/A"),
            "Link": metadata.get("link", "N/A"),
            "Duration": metadata.get("duration", "N/A"),
            "Remote Testing": metadata.get("remote_testing", "N/A"),
            "Adaptive Testing": metadata.get("adaptive_testing", "N/A"),
            "Language": metadata.get("language", "N/A")
        }
        matches.append(match)

    return matches







