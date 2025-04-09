import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec


file_path = os.path.abspath("products_catalogue.csv")
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df.columns = df.columns.str.strip()



pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


index_name = "assessments-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    return embedder.encode(text).tolist()


required_cols = [
    'Title', 'Link', 'Description', 'Remote_testing', 'Adaptive_Testing',
    'Job_levels', 'Language', 'Duration'
]

if all(col in df.columns for col in required_cols):
    vectors = []
    for index_num, row in df.iterrows():
        summary = f"{row['Title']} {row['Description']} {row['Job_levels']} {row['Language']} {row['Duration']}"
        embedding = embed_text(summary)
        vector = {
            "id": str(index_num),
            "values": embedding,
            "metadata": {
                "Assessment Name": row["Title"],
                "description": row["Description"],
                "link": row["Link"],
                "duration": row["Duration"],
                "job_level": row["Job_levels"],
                "remote_testing": row["Remote_testing"],
                "adaptive_testing": row["Adaptive_Testing"],
                "language": row["Language"]
            }
        }
        vectors.append(vector)


    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i+batch_size])

    print("✅ Data inserted into Pinecone successfully!")
else:
    print("❌ Error: Missing one or more required columns in the CSV.")
