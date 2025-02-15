import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import find_dotenv,load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from chunkers import chunk_pdf

import time

env_path = find_dotenv()
load_dotenv(env_path)
Pinecone_key = os.getenv("Pinecone_key")
def uplode_to_pinecone(index_name, docs, dimensions=None, embedding_model="bge-m3:latest"):
    pc = Pinecone(api_key=Pinecone_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        if dimensions is None:
            raise Exception("Please provide dimensions as an argument.")
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)  
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_documents(documents=docs)

    return True
    