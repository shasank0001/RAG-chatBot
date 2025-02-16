import streamlit as st
from chat import chat_with_ollama

st.title("Chat with Ollama")

model_name = st.text_input("Model Name", "llama3.2:1b")
question = st.text_area("Question", "What are the topics covered in this book?")

if st.button("Ask"):
    with st.spinner("Generating response..."):
        response = chat_with_ollama(model_name, question)
        st.write("Response:")
        st.write(response)