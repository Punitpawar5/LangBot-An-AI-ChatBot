import streamlit as st
from enhanced_backend import (
    chatbot, retrieve_all_threads, add_documents_to_rag, 
    get_conversation_summary, get_conversation_insights
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import pandas as pd
import time
# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Langbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Modern UI Styling ----------------
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #f4f9ff, #d6eaff);
        font-family: 'Segoe UI', sans-serif;
        color: #222;
    }
    .main-header {
        font-size: 2rem !important;
        font-weight: bold;
        color: #1976D2;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.8rem;
        animation: fadeIn 0.3s ease-in-out;
    }
    .chat-row.user {
        flex-direction: row-reverse;
    }
    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 0 0.6rem;
        flex-shrink: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        max-width: 75%;
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .assistant-message {
        background-color: #FFF8E1;
        border-left: 4px solid #FFB300;
        color: #5D4037;
    }
    .user-message {
        background-color: #E1F5FE;
        border-right: 4px solid #0288D1;
        color: #0D47A1;
    }
    .tool-message {
        background-color: #ECEFF1;
        border-left: 4px solid #607D8B;
        color: #263238;
        font-size: 0.85rem;
    }
    .typing-indicator {
        font-size: 0.85rem;
        color: #999;
        font-style: italic;
        margin-left: 42px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ---------------- Utility Functions ----------------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []
    st.session_state['pending_approval'] = None
    st.rerun()

def add_thread(thread_id):
    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except:
        return []

def format_message_for_display(msg):
    if isinstance(msg, HumanMessage):
        return {'role': 'user', 'content': msg.content, 'type': 'human'}
    elif isinstance(msg, AIMessage):
        return {'role': 'assistant', 'content': msg.content, 'type': 'ai'}
    else:
        return {'role': 'system', 'content': str(msg.content), 'type': 'other'}

def display_message(message):
    avatar_url = {
        'user': "https://i.imgur.com/z4d4kWk.png",
        'assistant': "https://i.imgur.com/3GvwNBf.png",
        'other': "https://i.imgur.com/tBrW8Zd.png"
    }.get(message['role'], "https://i.imgur.com/tBrW8Zd.png")
    bubble_class = {
        'user': "user-message",
        'assistant': "assistant-message",
        'other': "tool-message"
    }.get(message['role'], "tool-message")
    st.markdown(
    f"""
    <div class="chat-row {'user' if message['role'] == 'user' else ''}">
        <img src="{avatar_url}" class="chat-avatar">
        <div class="chat-message {bubble_class}">{message['content']}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Session State Init ----------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
if 'pending_approval' not in st.session_state:
    st.session_state['pending_approval'] = None
if 'conversation_mode' not in st.session_state:
    st.session_state['conversation_mode'] = 'normal'
if 'show_insights' not in st.session_state:
    st.session_state['show_insights'] = False
add_thread(st.session_state['thread_id'])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="main-header">LangBot 🤖</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An AI Chatbot</div>', unsafe_allow_html=True)
    st.subheader("Chat Management 💬")
    col1, col2 = st.columns(2)
    with col1:
        if st.button('New Chat', use_container_width=True):
            reset_chat()
    with col2:
        if st.button('Insights', use_container_width=True):
            st.session_state['show_insights'] = not st.session_state.get('show_insights', False)

    st.subheader("Mode Selection")
    mode = st.selectbox(
        "Choose conversation mode:",
        ['normal', 'rag', 'tool_use', 'advanced'],
        index=['normal', 'rag', 'tool_use', 'advanced'].index(st.session_state.get('conversation_mode', 'normal'))
    )
    st.session_state['conversation_mode'] = mode

    st.subheader("Knowledge Base (RAG)")
    uploaded_files = st.file_uploader("Upload documents for RAG", accept_multiple_files=True, type=['txt', 'pdf', 'md'])
    if uploaded_files and st.button('Add to Knowledge Base'):
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        try:
            doc_count = add_documents_to_rag(file_paths)
            st.success(f"Added {doc_count} document chunks to knowledge base!")
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
        import os
        for file_path in file_paths:
            try: os.remove(file_path)
            except: pass

    st.subheader("My Conversations")
    search_query = st.text_input("Search conversations 🔍")
    threads_to_show = st.session_state['chat_threads'][::-1]
    if search_query:
        threads_to_show = [tid for tid in threads_to_show if search_query.lower() in get_conversation_summary(tid).lower()]
    for i, thread_id in enumerate(threads_to_show[:10]):
        summary = get_conversation_summary(thread_id)
        is_current = thread_id == st.session_state['thread_id']
        button_label = f"{'🔵' if is_current else '⚪'} {summary}"
        if st.button(button_label, key=f"thread_{i}"):
            if thread_id != st.session_state['thread_id']:
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)
                st.session_state['message_history'] = [format_message_for_display(msg) for msg in messages]
                st.session_state['pending_approval'] = None
                st.rerun()

# ---------------- Main UI ----------------
st.markdown('<div class="main-header">Grind your thoughts with LangBot</div>', unsafe_allow_html=True)

if st.session_state.get('show_insights', False):
    with st.expander("Conversation Insights", expanded=True):
        insights = get_conversation_insights(st.session_state['thread_id'])
        if 'error' not in insights:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Messages", insights['total_messages'])
            col2.metric("Your Messages", insights['user_messages'])
            col3.metric("AI Messages", insights['ai_messages'])
            col4.metric("Sentiment", insights['sentiment'].title())

for message in st.session_state['message_history']:
    display_message(message)

user_input = st.chat_input('Type your message here... 💭')
if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input, 'type': 'human'})
    display_message({'role': 'user', 'content': user_input})

    config = {'configurable': {'thread_id': st.session_state['thread_id']}}
    with st.chat_message('assistant'):
        typing_placeholder = st.empty()
        typing_placeholder.markdown('<div class="typing-indicator">Gemini is thinking...</div>', unsafe_allow_html=True)
        full_response = ""
        try:
            for chunk, metadata in chatbot.stream({'messages': [HumanMessage(content=user_input)]}, config=config, stream_mode='messages'):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    typing_placeholder.markdown(full_response, unsafe_allow_html=False)
            typing_placeholder.markdown(full_response, unsafe_allow_html=False)
        except Exception as e:
            full_response = f"❌ Error: {str(e)}"
            typing_placeholder.markdown(full_response)
    st.session_state['message_history'].append({'role': 'assistant','content': full_response,'type': 'ai'})
    st.rerun()