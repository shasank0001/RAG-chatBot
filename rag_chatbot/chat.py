from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from query.queryRe_writting import re_write_query
from data_processing.DB import vector_store_retriever
import streamlit as st
def chat_with_doc(model_name, question):
    ollama = OllamaLLM(
        model=model_name,
        streaming=True
                       )
    output_parser = StrOutputParser()
    # context = re_write_query(question)
    retriever = vector_store_retriever("main-db", "Hands on Machine Learning with Scikit Learn and TensorFlow.pdf")
    context = retriever.invoke(question)

    prompt = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer with "I don't know".

    Context: {context}
    Question: {question}"""

    chat_prompt = ChatPromptTemplate.from_template(prompt)
    chat = (chat_prompt | ollama | output_parser)

    # Get the complete response
    response = chat.stream({"context": context, "question": question})
    return response

def basic_chat(model_name,quetion):
    
    ollama = OllamaLLM(
        model=model_name,
        streaming=True
        )
    output_parser = StrOutputParser()
    chat = (ollama | output_parser)
    response = chat.stream(quetion)

    return response
