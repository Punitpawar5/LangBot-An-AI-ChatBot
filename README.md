# LangBot: An AI ChatBot
This is an AI-powered multi-mode assistant integrating **Google Gemini** (free tier) for natural language understanding, **RAG (Retrieval-Augmented Generation)** for knowledge base queries, and various tool integrations for web search, weather lookup, and math calculations.  
It features a **modern Streamlit UI**, conversation history management, and analytics.

**Primary Use Cases:**
- General Q&A
- Searching custom knowledge bases(RAG)
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
  - Weather
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

## Demo Screenshots
Check out LinkedIn post about this project demo [here](https://www.linkedin.com/posts/punit-pawar5_ai-llm-langchain-activity-7361654454221234176-nIEJ?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD6bOdEBDSmX6bWWuAxedYglTGFE7pygkwU)

Demo images
![LangBot Chat UI](https://github.com/Punitpawar5/LangBot-An-AI-ChatBot/blob/main/Screenshot%20(256).png)
![Knowledge Base Search](https://github.com/Punitpawar5/LangBot-An-AI-ChatBot/blob/main/Screenshot%20(258).png)

## Installation

1. Clone the repo:

```bash
git clone https://github.com/Punitpawar5/LangBot-An-AI-ChatBot.git
cd LangBot-An-AI-ChatBot
```

2. Create a virtual environment:
   
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
3: Install dependencies:

```bash
pip install -r requirements.txt
```

4: Create .env file based on .env.example:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

Run the Streamlit app:
 ```bash
streamlit run enhanced_frontend.py
```
Open the browser at http://localhost:8501 and start chatting with the AI assistant.

## Architecture
```bash
Streamlit UI → LangGraph Chatbot → Gemini Model
                 ↓        ↑
               Tools (search, RAG, math, weather)
                 ↓
            External APIs / Vector DB
```

## Next Steps

Integrate a real weather API (e.g., OpenWeatherMap)

Replace DuckDuckGo with SerpAPI or Tavily for reliable web search

UI improvements: auto-scroll, dark mode, collapsible chat history

Voice mode: speech-to-text input, text-to-speech output

Deploy to Streamlit Cloud, Hugging Face Spaces, or a VPS
