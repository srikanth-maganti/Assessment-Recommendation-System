import pandas as pd
import chromadb
import os
from sentence_transformers import SentenceTransformer

file_path = os.path.abspath("products_catalogue.csv")
df=pd.read_csv(file_path,encoding='ISO-8859-1')

# Clean column names
df.columns = df.columns.str.strip()

# Initialize ChromaDB client
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="assessments")

# Initialize sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return embedder.encode(text).tolist()

# Required columns for processing
required_cols = ['Title', 'Link', 'Description', 'Remote_testing', 'Adaptive_Testing',
       'Job_levels', 'Language', 'Duration']

if all(col in df.columns for col in required_cols):
    for index, row in df.iterrows():
        summary=row["Title"]+" "+row["Description"]+" "+row["Job_levels"]+" "+row["Language"]+" "+row["Duration"]
        name=row["Title"]
        job_level=row["Job_levels"]
        description=row["Description"]
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
