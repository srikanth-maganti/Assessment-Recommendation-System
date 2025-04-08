import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_SERVER_URL = "https://chroma-server-umbu.onrender.com"


# Connect to ChromaDB server hosted on Render
db_client = chromadb.HttpClient(host=CHROMA_SERVER_URL)

# Get or create collection
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
find_matches("data scientist required")