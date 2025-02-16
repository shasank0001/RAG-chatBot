import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from chat import chat_with_ollama
from data_processing.DB import upload_to_pinecone
from data_processing.chunkers import chunk_pdf

st.title("Chat with Ollama")

# Section for chatting with Ollama
st.header("Chat with Ollama")
model_name = st.text_input("Model Name", "llama3.2:1b")
question = st.text_area("Question", "What are the topics covered in this book?")

if st.button("Ask"):
    with st.spinner("Generating response..."):
        response = chat_with_ollama(model_name, question)
        st.write("Response:")
        st.write(response)

# Section for uploading PDF to Pinecone
st.header("Upload PDF to Pinecone")
index_name = st.text_input("Index Name", "main-db")
namespace = st.text_input("Namespace", "default")
dimensions = st.number_input("Dimensions", min_value=1, value=768)
embedding_model = st.text_input("Embedding Model", "bge-m3:latest")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        # Process the uploaded PDF file
        docs = chunk_pdf(uploaded_file)
        # Upload the processed documents to Pinecone
        success = upload_to_pinecone(index_name, docs, namespace, dimensions, embedding_model)
        if success:
            st.success("PDF uploaded to Pinecone successfully!")
        else:
            st.error("Failed to upload PDF to Pinecone.")