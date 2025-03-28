import sys
import os
import PyPDF2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from chat import chat_with_doc
from chat import basic_chat , chat_with_code ,chat_with_reasoning_model
from data_processing.DB import upload_to_pinecone
from data_processing.chunkers import chunk_pdf
from langchain_community.document_loaders.pdf import PyPDFLoader


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
    <style>
    /* Modern dark theme */
    :root {
        --main-bg: #0f0f0f;
        --secondary-bg: #1f1f1f;
        --accent: #2d5af7;
        --text: #e0e0e0;
        --border: #2d2d2d;
    }
    
    .stApp {
        background: var(--main-bg);
    }
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    /* User message specific */
    [data-testid="stChatMessage"][data-testid="user"] {
        background: var(--accent);
        border: none;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(transparent, var(--main-bg) 20%);
        padding: 2rem 3rem;
        z-index: 100;
    }
    
    .stChatInput input {
        background: var(--secondary-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        color: var(--text) !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }
    
    .stChatInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(45,90,247,0.2) !important;
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: var(--secondary-bg);
        border-right: 1px solid var(--border);
        padding: 2rem 1rem;
    }
    
    /* Buttons */
    .stButton button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(45,90,247,0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px dashed var(--border);
    }
    
    /* Selectbox */
    [data-testid="stSelectbox"] {
        background: var(--secondary-bg);
        border-radius: 8px;
        border-color: var(--border);
    }
    
    /* Text elements */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }
    
    /* Hide default elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Loading animation */
    .stSpinner {
        border-color: var(--accent) !important;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

with st.sidebar:
    st.title("RAG Chatbot")
    st.markdown("---")
    chat_mode = st.radio("Select Chat Mode", ["Basic Chat", "Chat with Document","code Assistant","Reasoning Assistant"])
    # Upload section
    st.header("📁 Document Upload")
    index_name = "main-db"
    namespace = st.text_input("Namespace", "default")
    dimensions = 1024
    embedding_model = "bge-m3:latest"
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
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


file_content = ""
uploaded_file = st.file_uploader(
    "Upload a File",
    type=["py", "js", "java", "cpp", "c", "cs", "rb", "go", "php", "html", "css", "json", "xml", "txt", "pdf"],
    label_visibility="collapsed"
)
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".pdf"):
            # Handle PDF files
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
            st.success(f"📄 PDF file '{uploaded_file.name}' uploaded and processed successfully!")
        else:
            # Handle other text-based files
            file_content = uploaded_file.read().decode("utf-8")
            st.success(f"📄 File '{uploaded_file.name}' uploaded successfully!")
    except UnicodeDecodeError:
        try:
            # Fallback to reading the file with 'latin-1' encoding for non-PDF files
            file_content = uploaded_file.read().decode("latin-1")
            st.warning(f"📄 File '{uploaded_file.name}' uploaded with fallback encoding (latin-1).")
        except Exception as e:
            st.error(f"❌ Failed to read the file: {str(e)}")
    except Exception as e:
        st.error(f"❌ Failed to process the file: {str(e)}")
col1, col2 = st.columns([2, 1])
with col1:

    # Main chat interface
    st.title("💬 AI Assistant")

    # Create a container for messages with some bottom padding for input
    message_container = st.container()
        
    # Add padding at the bottom to prevent chat input from overlapping messages
    st.markdown("<div style='padding-bottom: 100px'></div>", unsafe_allow_html=True)

    with message_container:
        for message in st.session_state.messages:
            with st.chat_message(
                message["role"],
                avatar="🤖" if message["role"] == "assistant" else "👤"
            ):
                st.markdown(message["content"])


    # Handle new messages
    if prompt := st.chat_input("Message AI Assistant...", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            if chat_mode == "Chat with Document":
                response_generator = chat_with_doc(model_name, prompt)
            elif chat_mode == "Basic Chat":
                response_generator = basic_chat(model_name, prompt + file_content)
                file_content = ""
            elif chat_mode == "code Assistant":
                response_generator = chat_with_code( prompt , file_content)
                file_content = ""
            elif chat_mode == "Reasoning Assistant":
                print("im in")
                response_generator = chat_with_reasoning_model(prompt,file_content)
                file_content = ""

            for chunk in response_generator:
                if isinstance(chunk, str):  # Ensure the chunk is a string
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")  # Show typing effect
        
            message_placeholder.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
        
        st.rerun()
        
with col2:
    st.header("📊 Chat Information")
    st.info(f"""
    - Model: {model_name}
    - Messages: {len(st.session_state.messages)}
    - Index: {index_name}
    """)
