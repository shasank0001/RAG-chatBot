import base64
import streamlit as st
from langchain_ollama import OllamaLLM  # make sure you're using the latest version

st.title("Chatbot with Ollama & LangChain")

# Initialize conversation history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: optional image uploader
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Main text input for chat
user_message = st.text_input("Enter your message:")

if st.button("Send"):
    # Build prompt with image data if available
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        # Encode the image to base64 (for demonstration, we take a short preview)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        image_preview = encoded_image[:50]  # preview only first 50 characters
        prompt = f"{user_message}\n[Attached Image Data (base64 preview): {image_preview}...]"
    else:
        prompt = user_message

    # Add user input to conversation history
    st.session_state.history.append({"role": "user", "message": prompt})

    # Instantiate the Ollama chat model via LangChain.
    # (Ensure your local Ollama server is running and adjust base_url as needed.)
    chat = OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434")

    # For simplicity, concatenate conversation history into one prompt.
    conversation = "\n".join([f"{entry['role']}: {entry['message']}" for entry in st.session_state.history])
    
    # Get the assistant's response from the model.
    response = chat.invoke(conversation)
    st.session_state.history.append({"role": "assistant", "message": response})
    
    # Display the response.
    st.write("**Assistant:**", response)

# Display the full conversation history
st.markdown("### Conversation History")
for entry in st.session_state.history:
    st.write(f"**{entry['role'].capitalize()}:** {entry['message']}")
