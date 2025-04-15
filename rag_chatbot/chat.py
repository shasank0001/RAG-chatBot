from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from query.queryRe_writting import re_write_query
from data_processing.DB import vector_store_retriever
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()


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

def chat_with_code(question, code):
    model_name = "deepcoder:1.5b-preview-q8_0"
    ollama = OllamaLLM(
        model=model_name,
        streaming=True
    )
    output_parser = StrOutputParser()
    prompt = """
    You are a helpful coding assistant. If given any code, try to help using it. If nothing is given, help according to the question.
    Code: {code}
    Question: {question}
    """
    chat_prompt = ChatPromptTemplate.from_template(prompt)

    chat = (chat_prompt | ollama | output_parser)
    # Pass inputs as a dictionary
    response = chat.stream({"code": code, "question": question})

    return response


def chat_with_reasoning_model(question, file_content):
    model_name = "exaone-deep:2.4b-q8_0" #deepseek-r1:14b-qwen-distill-q8_0
    ollama = OllamaLLM(
        model=model_name,
        streaming=True
    )
    output_parser = StrOutputParser()
    prompt = """
    You are a helpful research assistant. If any file content is provided help based on thet or else try helping using only Question
    file_content: {file_content}
    Question: {question}
    """
    chat_prompt = ChatPromptTemplate.from_template(prompt)

    chat = (chat_prompt | ollama | output_parser)
    # Pass inputs as a dictionary
    response = chat.stream({"file_content":file_content , "question": question})

    return response


def chat_with_gemini(pre_chat, prompt, context="", image=None):
    os.getenv("GOOGLE_API_KEY")
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
            # Use regular gemini for text-only processing
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
        output_parser = StrOutputParser()
        
        template = """
            past_chat: {pre_chat}
            Context: {context}
            Question: {prompt}
            
            Please provide a helpful response based on the past_chat, context and question above.
            If no context is provided, just answer the question directly.
            """
            
        chat_prompt = ChatPromptTemplate.from_template(template)
        chain = chat_prompt | llm | output_parser
        
        response = chain.invoke({
                "context": context,
                "prompt": prompt,
                "pre_chat": pre_chat
            })
            
        return response
        
    except Exception as e:
        return f"Error: {str(e)}" 
