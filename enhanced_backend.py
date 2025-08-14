from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import sqlite3
import requests
import json
from datetime import datetime
import os
from pathlib import Path

# ---------------- Environment & API Key ----------------
load_dotenv()
# Prefer GEMINI_API_KEY but also support GOOGLE_API_KEY (used by google-generativeai)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # langchain_google_genai reads this
else:
    # Graceful fallback; LangChain will error on first call, but we keep module importable
    print("⚠️  No GEMINI_API_KEY/GOOGLE_API_KEY found. Set it in your .env to enable Gemini.")

# ---------------- LLM & Embeddings (Gemini Free Tier) ----------------
# Chat model: gemini-1.5-flash (fast & in free tier)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
# Embedding model: models/embedding-001 (free tier)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ---------------- Vector Store (Chroma) ----------------
VECTOR_DB_PATH = "./chroma_db"
if not os.path.exists(VECTOR_DB_PATH):
    os.makedirs(VECTOR_DB_PATH)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

# ---------------- Document processing for RAG ----------------
def load_documents(file_paths: List[str]):
    """Load and process documents for RAG"""
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    if splits:
        vectorstore.add_documents(splits)
    return len(splits)

# ---------------- Tools ----------------
@tool
def search_web(query: str) -> str:
    """Search the web for current information"""
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (placeholder; integrate a real API for production)"""
    try:
        return f"Weather in {city}: Sunny, 25°C (placeholder — integrate a real weather API for production)"
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"

@tool
def calculate_math(expression: str) -> str:
    """Calculate mathematical expressions safely"""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Calculation failed: {str(e)}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base using RAG"""
    try:
        docs = vectorstore.similarity_search(query, k=3)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            return f"Knowledge base results:\n{context}"
        else:
            return "No relevant information found in knowledge base"
    except Exception as e:
        return f"Knowledge base search failed: {str(e)}"

# List of available tools
tools = [search_web, get_weather, calculate_math, search_knowledge_base]
tool_node = ToolNode(tools)

# ---------------- Enhanced State ----------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retry_count: int
    human_feedback: Optional[str]
    requires_human_approval: bool
    conversation_mode: str  # "normal", "rag", "tool_use"

# ---------------- System Prompt ----------------
SYSTEM_PROMPT = """You are an advanced AI assistant with access to various tools and a knowledge base. 

Your capabilities include:
1. General conversation and assistance
2. Web search for current information
3. Weather information lookup
4. Mathematical calculations
5. Knowledge base search using RAG

Guidelines:
- Use tools when appropriate to provide accurate, up-to-date information
- For factual questions, prefer searching the knowledge base first, then web search if needed
- Be helpful, accurate, and conversational
- If you're unsure about something sensitive or important, ask for human approval
- Always explain your reasoning when using tools

Available tools: {tool_names}
"""

def should_continue(state: ChatState) -> str:
    """Decide whether to continue with tools, get human approval, or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # Check if human approval is required
    if state.get("requires_human_approval", False):
        return "human_approval"

    # Check if we have tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Check retry logic
    if state.get("retry_count", 0) > 0:
        return "retry"

    return "end"

def chat_node(state: ChatState) -> ChatState:
    """Main chat processing node"""
    messages = state["messages"]
    retry_count = state.get("retry_count", 0)

    # Create system message with tool information
    tool_names = [tool.name for tool in tools]
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(tool_names=", ".join(tool_names)))

    # Prepare messages for LLM
    llm_messages = [system_message] + messages

    # Bind tools to LLM (Gemini supports tool/function calling via LangChain)
    llm_with_tools = llm.bind_tools(tools)

    try:
        response = llm_with_tools.invoke(llm_messages)

        # Check if response requires human approval (example criteria)
        content = (response.content or "").lower()
        sensitive_topics = ["delete", "remove", "harmful", "dangerous"]
        requires_approval = any(topic in content for topic in sensitive_topics)

        return {
            "messages": [response],
            "retry_count": 0,
            "requires_human_approval": requires_approval
        }

    except Exception as e:
        error_message = AIMessage(content=f"I encountered an error: {str(e)}. Let me try again.")
        return {
            "messages": [error_message],
            "retry_count": retry_count + 1,
            "requires_human_approval": False
        }

def retry_node(state: ChatState) -> ChatState:
    """Handle retry logic"""
    retry_count = state.get("retry_count", 0)

    if retry_count >= 3:
        error_message = AIMessage(
            content="I'm having trouble processing your request after multiple attempts. Could you please rephrase or try a different question?"
        )
        return {
            "messages": [error_message],
            "retry_count": 0,
            "requires_human_approval": False
        }

    # Try again with simplified approach (still Gemini, lower temp)
    simple_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    try:
        response = simple_llm.invoke(state["messages"][-5:])  # Use only recent messages
        return {
            "messages": [response],
            "retry_count": 0,
            "requires_human_approval": False
        }
    except Exception as e:
        return {
            "messages": [],
            "retry_count": retry_count + 1,
            "requires_human_approval": False
        }

def human_approval_node(state: ChatState) -> ChatState:
    """Handle human-in-the-loop approval"""
    approval_message = AIMessage(
        content="⚠️ This response requires human approval. Please review and approve before proceeding."
    )

    return {
        "messages": [approval_message],
        "retry_count": 0,
        "requires_human_approval": False,
        "human_feedback": "pending_approval"
    }

def process_human_feedback(state: ChatState, feedback: str) -> ChatState:
    """Process human feedback"""
    if feedback.lower() in ["approve", "yes", "ok"]:
        feedback_message = AIMessage(content="✅ Approved by human. Proceeding with the response.")
    else:
        feedback_message = AIMessage(content="❌ Not approved. Please provide alternative instructions.")

    return {
        "messages": [feedback_message],
        "human_feedback": feedback,
        "requires_human_approval": False
    }

# ---------------- Database & Checkpointer ----------------
conn = sqlite3.connect(database='enhanced_chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ---------------- Build the graph ----------------
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("retry", retry_node)
graph.add_node("human_approval", human_approval_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    should_continue,
    {
        "tools": "tools",
        "human_approval": "human_approval",
        "retry": "retry",
        "end": END
    }
)
graph.add_edge("tools", "chat_node")
graph.add_edge("retry", "chat_node")
graph.add_edge("human_approval", END)

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)

# ---------------- Utility functions ----------------
def retrieve_all_threads():
    """Retrieve all conversation threads"""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        all_threads.add(thread_id)
    return list(all_threads)

def add_documents_to_rag(file_paths: List[str]) -> int:
    """Add documents to RAG knowledge base"""
    return load_documents(file_paths)

def get_conversation_summary(thread_id: str) -> str:
    """Get a summary of a conversation thread"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])

        if not messages:
            return "Empty conversation"

        # Create a simple summary from first user message
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        if user_messages:
            first_msg = user_messages[0].content[:50]
            return f"{first_msg}..." if len(user_messages[0].content) > 50 else first_msg

        return "No user messages"
    except Exception as e:
        return f"Error loading conversation: {str(e)}"

# ---------------- Advanced conversation features ----------------
def analyze_conversation_sentiment(messages: List[BaseMessage]) -> str:
    """Analyze conversation sentiment (very simple heuristic)"""
    positive_words = ["good", "great", "excellent", "happy", "pleased", "satisfied"]
    negative_words = ["bad", "terrible", "awful", "sad", "angry", "disappointed"]

    text = " ".join([getattr(msg, 'content', '').lower() for msg in messages])

    positive_score = sum(1 for word in positive_words if word in text)
    negative_score = sum(1 for word in negative_words if word in text)

    if positive_score > negative_score:
        return "positive"
    elif negative_score > positive_score:
        return "negative"
    else:
        return "neutral"

def get_conversation_insights(thread_id: str) -> dict:
    """Get insights about a conversation"""
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])

        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "sentiment": analyze_conversation_sentiment(messages),
            "has_tool_usage": any(hasattr(msg, 'tool_calls') and msg.tool_calls for msg in ai_messages),
            "conversation_length": sum(len(getattr(msg, 'content', '')) for msg in messages)
        }
    except Exception as e:
        return {"error": str(e)}
