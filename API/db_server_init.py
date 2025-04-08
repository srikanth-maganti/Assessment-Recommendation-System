import pandas as pd
import chromadb
import os
from sentence_transformers import SentenceTransformer

# ChromaDB Render server URL
CHROMA_SERVER_URL = "https://chroma-server-umbu.onrender.com"  # no https and no trailing slash

file_path = os.path.abspath("products_catalogue.csv")
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Clean column names
df.columns = df.columns.str.strip()

# Connect to ChromaDB server hosted on Render
db_client = chromadb.HttpClient(host=CHROMA_SERVER_URL)

# Get or create collection
collection = db_client.get_or_create_collection(name="assessments")

# Check if collection already has records
existing_ids = collection.count()
if existing_ids > 0:
    print("✅ Collection already populated. Skipping data re-initialization.")
else:
    # Initialize embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_text(text):
        return embedder.encode(text).tolist()

    required_cols = [
        'Title', 'Link', 'Description', 'Remote_testing', 'Adaptive_Testing',
        'Job_levels', 'Language', 'Duration'
    ]

    if all(col in df.columns for col in required_cols):
        for index, row in df.iterrows():
            summary = f"{row['Title']} {row['Description']} {row['Job_levels']} {row['Language']} {row['Duration']}"
            name = row["Title"]
            job_level = row["Job_levels"]
            description = row["Description"]
            link = row["Link"]
            duration = row["Duration"]
            remote = row["Remote_testing"]
            adaptive = row["Adaptive_Testing"]
            language = row["Language"]

            embedding = embed_text(summary)

            collection.add(
                embeddings=[embedding],
                documents=[name],
                metadatas=[{
                    "link": link,
                    "duration": duration,
                    "job_level": job_level,
                    "description": description,
                    "remote_testing": remote,
                    "adaptive_testing": adaptive,
                    "language": language
                }],
                ids=[str(index)]
            )

        print("✅ All assessment catalogue data successfully added to ChromaDB!")
    else:
        print("❌ Error: Missing one or more required columns in the CSV.")
