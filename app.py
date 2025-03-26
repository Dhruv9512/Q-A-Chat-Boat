import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from langchain.memory import ConversationBufferMemory
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv () 

# Create model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0.6,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)


# Create memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
memory = st.session_state.memory

# Initialize the System Message in memory
# You are a Professor who has all IT field knowledge
memory.chat_memory.add_message(SystemMessage(content="You are a helpful assistant."))


# Frontend UI using streamlit
st.set_page_config(page_title="Q&A Chat Boat")

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #F7F8FC;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin: 5px 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #6C63FF;
        color: white;
        align-self: flex-end;
    }
    .ai-message {
        background-color: #E8EAF6;
        color: black;
        align-self: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🤖 Q&A Chatbot</h1>", unsafe_allow_html=True)

# Using a form to group text input and button
with st.form(key="chat_form"):
    user_input = st.text_input("💬 Ask me anything...", key="input")
    submit_button = st.form_submit_button("Send")
# Handle input after clicking the button
if submit_button and user_input:

    # set human message
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # loading till the ans predicted
    with st.spinner("🤖 Thinking..."):
        time.sleep(3) 
        response = model.invoke(memory.chat_memory.messages)
    
    # set ai message
    memory.chat_memory.add_message(AIMessage(content=str(response.content)))

    st.write(response.content)
    