from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables for LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template for Mental Health Assistant
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a mental health assistant. Please respond to the user queries with empathy, understanding, and support."),
    ("user", "Question: {question}")
])

# Streamlit Web Interface
st.set_page_config(page_title="Mental Health Assistance Chatbot", page_icon="🧠", layout="wide")

# Header for the page
st.title('Mental Health Assistance Chatbot')
st.subheader("Welcome to your personal mental health assistant. I'm here to help you.")

# Instructions on how to use the chatbot
st.markdown("""
    - **Ask me anything about mental health.**
    - **I will respond with advice, empathy, and understanding.**
    - **Feel free to type 'exit' anytime to stop the conversation.**
""")

# User Input Section
input_text = st.text_area("How are you feeling today?", height=100, max_chars=500)

# Initialize LLM with Ollama
llm = Ollama(model="llama3.2")

# Create the LLM Chain with the prompt and model
chain = LLMChain(prompt=prompt, llm=llm)

# Chat History Section
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to display chat history
def display_chat_history():
    for chat in st.session_state.chat_history:
        st.markdown(f"**{chat['role']}**: {chat['message']}")

# Display previous chat history
display_chat_history()

# Handle user query and provide response
if input_text:
    if input_text.lower() == "exit":
        st.session_state.chat_history.append({"role": "system", "message": "Goodbye! Take care."})
        st.session_state.chat_history = []  # Clear the chat history
        st.write("Goodbye! Stay safe and take care of yourself.")
    else:
        try:
            # Run the chain with the user's question
            response = chain.run({"question": input_text})
            # Display the response
            st.session_state.chat_history.append({"role": "user", "message": input_text})
            st.session_state.chat_history.append({"role": "assistant", "message": response})
            display_chat_history()  # Show updated chat history
        except Exception as e:
            st.error(f"An error occurred: {e}")

