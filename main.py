import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from streamlit_chat import message
from huggingface_hub import InferenceClient
from chatbot_backend import EmotionDetector, AssistantResponder


# Function to initialize conversation memory
def initialize_memory():
    return ConversationBufferWindowMemory(k=3, return_messages=True)

# Load environment variables
dotenv_path = '.env'
load_dotenv(dotenv_path)
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_ENDPOINT_URL = os.getenv("API_ENDPOINT_URL")

# Initialize emotion detector and assistant responder
emotion_detector = EmotionDetector(HUGGINGFACE_API_TOKEN, API_ENDPOINT_URL)
assistant_responder = AssistantResponder(HUGGINGFACE_API_TOKEN, API_ENDPOINT_URL)

# App setup
st.set_page_config(page_title="Unified Bot Management", layout="wide")

# Sidebar
st.sidebar.title("Unified Bot Management")

if 'selected_bot_name' not in st.session_state:
    st.session_state['selected_bot_name'] = "Llama2 Bot"

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = initialize_memory()

# Main chat area
chat_container = st.container()
with chat_container:
    st.title(f"Chat with {st.session_state['selected_bot_name']}")
    
    # Bot parameters
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    
    response_container = st.container()
    textcontainer = st.container()
    
    with textcontainer:
        query = st.text_input("Type your message here: ", key="input")
        if query:
            memory_dict = st.session_state['buffer_memory'].load_memory_variables({})
            memory = " ".join([msg.content for msg in memory_dict['history']])
            response = assistant_responder.generate_response(query, memory, temperature, top_p, max_length)
            st.session_state['requests'].append(query)
            st.session_state['responses'].append(response)
            st.session_state['buffer_memory'].save_context({"input": query}, {"output": response})
            
            # Detect emotion
            emotion = emotion_detector.detect_emotion(query)
            st.sidebar.write(f"Detected emotion: {emotion}")
    
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')

# Sidebar button to clear chat history
def clear_chat_history():
    st.session_state['responses'] = ["How can I assist you?"]
    st.session_state['requests'] = []
    st.session_state['buffer_memory'] = initialize_memory()
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
