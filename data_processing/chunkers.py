from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def chunk_pdf(pdf_path):
    #Recursive Character Text Splitting 
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap=80) # ["\n\n", "\n", " ", ""] 65,450
    pages = text_splitter.split_documents(docs)
    
    return pages
    

