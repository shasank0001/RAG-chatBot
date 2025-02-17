import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from chat import chat_with_ollama
from data_processing.DB import upload_to_pinecone
from data_processing.chunkers import chunk_pdf
from langchain_community.document_loaders.pdf import PyPDFLoader

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF upload and configuration
with st.sidebar:
    st.header("Upload PDF")
    index_name = st.text_input("Index Name", "main-db")
    namespace = st.text_input("Namespace", "default")
    dimensions = st.number_input("Dimensions", min_value=1, value=768)
    embedding_model = st.text_input("Embedding Model", "bge-m3:latest")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file to a temporary location
            with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            pdf_path = os.path.join("/tmp", uploaded_file.name)
            docs = chunk_pdf(pdf_path)
            
            success = upload_to_pinecone(index_name, docs, namespace, dimensions, embedding_model)
            if success:
                st.success("PDF uploaded to Pinecone successfully!")
            else:
                st.error("Failed to upload PDF to Pinecone.")

# Main chat interface
st.title("Chat with Ollama")
model_name = st.text_input("Model Name", "llama3.2:1b")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about the document?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_with_ollama(model_name, prompt)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})