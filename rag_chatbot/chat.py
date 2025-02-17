from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from query.queryRe_writting import re_write_query
import streamlit as st
def chat_with_ollama(model_name, question):
    ollama = OllamaLLM(model=model_name)
    output_parser = StrOutputParser()
    context = re_write_query(question)

    prompt = """Answer the question based on the context below. If the
    question cannot be answered using the information provided answer with "I don't know".

    Context: {context}
    Question: {question}"""

    chat_prompt = ChatPromptTemplate.from_template(prompt)
    chat = (chat_prompt | ollama | output_parser)

    # Get the complete response
    response = chat.invoke({"context": context, "question": question})
    return response