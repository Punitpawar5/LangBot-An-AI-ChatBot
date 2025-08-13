# LangBot: An AI ChatBot
This is an AI-powered multi-mode assistant integrating **Google Gemini** (free tier) for natural language understanding, **RAG (Retrieval-Augmented Generation)** for knowledge base queries, and various tool integrations for web search, weather lookup, and math calculations.  
It features a **modern Streamlit UI**, conversation history management, and analytics.

**Primary Use Cases:**
- General Q&A
- Searching custom knowledge bases
- Performing web searches
- Answering math problems
- Maintaining multiple conversation threads

---

## Features

### Frontend
- Modern, responsive chat layout
- User & AI message styling
- Animated typing indicator
- Conversation insights (metrics + sentiment analysis)
- Multiple conversation management
- Knowledge base document upload for RAG

### Backend
- Multi-node LangGraph flow:
  - Chat node (Gemini)
  - Tool execution
  - Retry node
  - Human-in-the-loop approval
- Custom tools:
  - Web Search (DuckDuckGo)
  - Weather (placeholder)
  - Math Calculator
  - Knowledge Base Search (Chroma + Embeddings)
- Document ingestion & chunking for RAG

---

## Tech Stack
- **Frontend:** Streamlit, HTML/CSS, Markdown Rendering
- **Backend:** Python, LangChain, LangGraph, Google Gemini API, ChromaDB, PyPDF, TextLoader
- **Tools:** DuckDuckGoSearchRun, Math evaluator
- **Environment:** `python-dotenv`

---
