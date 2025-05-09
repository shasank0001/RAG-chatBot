import sys
import os
import PyPDF2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import google.generativeai as genai  


import streamlit as st
from chat import chat_with_doc
from chat import basic_chat , chat_with_code ,chat_with_reasoning_model
from data_processing.DB import upload_to_pinecone
from data_processing.chunkers import chunk_pdf
from langchain_community.document_loaders.pdf import PyPDFLoader
from fpdf import FPDF
from docx import Document
import markdown
import io
from PIL import Image

def export_chat_to_pdf(messages, filename="chat_history.pdf"):
    def clean_text(text):
        # Replace emojis and special characters with their text equivalents or remove them
        emoji_mapping = {
            '🤖': '[BOT]',
            '👤': '[USER]',
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '📄': '[FILE]',
            '💾': '[SAVE]',
            '📥': '[DOWNLOAD]',
            '⚠️': '[WARNING]'
        }
        
        # Replace known emojis with their text versions
        for emoji, replacement in emoji_mapping.items():
            text = text.replace(emoji, replacement)
            
        # Remove any remaining non-latin1 characters
        return ''.join(char for char in text if ord(char) < 256)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_text_color(0, 0, 0)  # Black text color
    pdf.cell(200, 10, txt="Chat History", ln=True, align="C")
    pdf.ln(10)  # Add a line break

    for message in messages:
        role = "Assistant" if message["role"] == "assistant" else "User"
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt=f"{role}:", ln=True)
        pdf.set_font("Arial", size=12)
        
        # Clean the message content before adding to PDF
        cleaned_content = clean_text(message["content"])
        pdf.multi_cell(0, 10, txt=cleaned_content)
        pdf.ln(5)  # Add some spacing between messages

    pdf.output(filename)
    return filename


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
    
    /* Background image will be added dynamically when uploaded */
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Transparent Sidebar with glass effect */
    [data-testid="stSidebar"] {
        background-color: rgba(31, 31, 31, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 2rem 1rem;
    }
    
    /* Add slight glow to sidebar elements for better readability */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stText, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        text-shadow: 0 0 5px rgba(0,0,0,0.5);
    }
    
    /* Style sidebar widgets to match transparent theme */
    [data-testid="stSidebar"] [data-testid="stRadio"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div:first-child,
    [data-testid="stSidebar"] [data-testid="stTextInput"] > div:first-child {
        background-color: rgba(45, 45, 45, 0.6) !important;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* File uploader styling to match transparent theme */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: rgba(45, 45, 45, 0.6) !important;
        border-radius: 12px;
        padding: 1rem;
        border: 1px dashed rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Button styling in sidebar */
    [data-testid="stSidebar"] .stButton button {
        background: rgba(45, 90, 247, 0.8) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
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
    chat_mode = st.radio("Select Chat Mode", ["Basic Chat", "Chat with Document", "code Assistant", "Reasoning Assistant"])
    
    # Background image upload
    st.header("🖼️ Background Image")
    
    # Default background options
    default_bg_option = st.selectbox(
        "Choose a default background:",
        ["None", "Gradient Dark", "Blue Abstract", "Matrix Code", "Starry Night"]
    )
    
    # Apply default background based on selection
    if default_bg_option != "None":
        if default_bg_option == "Gradient Dark":
            bg_url = "https://images.unsplash.com/photo-1557682250-33bd709cbe85"
        elif default_bg_option == "Blue Abstract":
            bg_url = "https://images.unsplash.com/photo-1579546929518-9e396f3cc809"
        elif default_bg_option == "Matrix Code":
            bg_url = "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5"
        elif default_bg_option == "Starry Night":
            bg_url = "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a"
            
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url({bg_url}) !important;
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(15, 15, 15, 0.7);
                z-index: -1;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    # Custom background upload
    st.write("Or upload your own:")
    bg_image = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
    if bg_image is not None:
        # Convert the uploaded image to base64
        import base64
        from io import BytesIO
        
        image = Image.open(bg_image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Set the background image using custom CSS with !important to override other styles
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{img_str}) !important;
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}
            /* Ensure the overlay doesn't block the background image */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(15, 15, 15, 0.7);
                z-index: -1;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
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
    model_type = st.selectbox(
        "Select Model",
        ["Air","Pro","Max","STEM","cloud"],
        index=0
    )
    if model_type == "Air":
        model_name = "gemma3:4b" # change this to the actual model name
    elif model_type == "Pro":
        model_name = "gemma3:12b-it-q8_0"
    elif model_type == "Max":
        model_name = "gemma3:27b-it-q8_0"
    elif model_type == "cloud":
        model_name = "gemini-2.0-flash-001"
    elif model_type == "STEM":
        model_name = "phi4:14b-q8_0"

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


with st.sidebar:
    if st.button("💾 Save Chat as PDF"):
        if st.session_state.messages:
            with st.spinner("Saving chat to PDF..."):
                pdf_file = export_chat_to_pdf(st.session_state.messages)
                st.success("✅ Chat saved as PDF!")
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Chat PDF",
                        data=f,
                        file_name="chat_history.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("⚠️ No chat messages to save!")

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
            
            if model_type == "cloud":
                response_generator = chat_with_gemini(st.session_state.messages, prompt, file_content, None)
                print("Gemini response generator:", response_generator)
                full_response = response_generator
                file_content = ""
            else:
                if chat_mode == "Chat with Document":
                    response_generator = chat_with_doc(model_name, prompt)
                elif chat_mode == "Basic Chat":
                    response_generator = basic_chat(model_name, prompt + file_content)
                    file_content = ""
                elif chat_mode == "code Assistant":
                    response_generator = chat_with_code(prompt, file_content)
                    file_content = ""
                elif chat_mode == "Reasoning Assistant":
                    response_generator = chat_with_reasoning_model(prompt, file_content)
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
        print(st.session_state.messages)
        
        st.rerun()

