from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def clean_context(context):
    templet = """ The draft response does not contain any duplicate content. """
    