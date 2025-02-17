import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from chat import chat_with_ollama
from data_processing.DB import upload_to_pinecone
from data_processing.chunkers import chunk_pdf
from langchain_community.document_loaders.pdf import PyPDFLoader

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

# Sidebar configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/streamlit_app_demo/example/logo.png", width=100)
    st.title("RAG Chatbot")
    st.markdown("---")
    
    # Upload section
    st.header("📁 Document Upload")
    index_name = st.text_input("Index Name", "main-db")
    namespace = st.text_input("Namespace", "default")
    dimensions = st.number_input("Dimensions", min_value=1, value=1024)
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
                st.success("✅ PDF uploaded successfully!")
            else:
                st.error("❌ Upload failed")
    
    st.markdown("---")
    model_name = st.selectbox(
        "Select Model",
        ["llama3.2:1b"],
        index=0
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = chat_with_ollama(model_name, prompt)
                message_placeholder.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.header("📊 Chat Information")
    st.info(f"""
    - Model: {model_name}
    - Messages: {len(st.session_state.messages)}
    - Index: {index_name}
    """)