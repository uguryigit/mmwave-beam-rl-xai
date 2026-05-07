import os
import boto3
import docx
import ollama
import json
from pinecone import Pinecone, ServerlessSpec

# AWS credentials are read from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env
# vars (or ~/.aws/credentials, or IAM role). See .env.example.
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
)
# --- 1. Function to get embeddings from ollama ---
def get_embedding(text):
        """Get embeddings from AWS Bedrock Titan v2"""
        body = {"inputText": text}
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

# --- 2. Read Word document ---
def read_docx(file_path):
    doc = docx.Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs

# --- 3. Connect to Pinecone ---
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)

index_name = "dqn-explain"
dimension = 1024   # must match embedding size

# --- 4. Create index if it doesn’t exist ---
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",   # cosine similarity for semantic search
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# --- 5. Process document and upsert ---
file_path = "Simulation Setup RAG.docx"
paragraphs = read_docx(file_path)

vectors = []
for i, para in enumerate(paragraphs):
    emb = get_embedding(para)
    vectors.append({
        "id": str(i),
        "values": emb,
        "metadata": {
            "title": "Simulation Setup",
            "content": para
        }
    })

# Upsert to Pinecone
index.upsert(vectors)

print("All document paragraphs indexed into Pinecone!")
