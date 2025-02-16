from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.DB import vector_store_retriever

def re_write_query(query):
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database of text books and pdfs. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {query}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | OllamaLLM(model = "qwen2.5:7b-instruct") 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retriever = vector_store_retriever("main-db", "Hands on Machine Learning with Scikit Learn and TensorFlow.pdf")

    # Retrieve
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"query": query})
    
    return docs

def get_unique_union(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


