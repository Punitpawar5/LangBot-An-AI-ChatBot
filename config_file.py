# config.py - Configuration file for Enhanced Chatbot (Gemini version)

import os
from typing import Dict, List, Optional

class ChatbotConfig:
    """Configuration class for the enhanced chatbot (Google Gemini API)"""
    
    # API Keys (set these in your .env file)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    WEATHER_API_KEY: Optional[str] = os.getenv("WEATHER_API_KEY")  # Optional
    
    # Database Settings
    DATABASE_PATH: str = "enhanced_chatbot.db"
    VECTOR_DB_PATH: str = "./chroma_db"
    
    # LLM Settings
    LLM_MODEL: str = "gemini-1.5-flash"  # Free tier fast model
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_DOCS: int = 3
    
    # Retry Settings
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_MODEL: str = "gemini-1.5-flash"  # Use same model for retry
    
    # HITL Settings
    SENSITIVE_KEYWORDS: List[str] = [
        "delete", "remove", "harmful", "dangerous", "illegal",
        "hack", "break", "destroy", "attack"
    ]
    AUTO_APPROVAL_THRESHOLD: float = 0.8  # Confidence threshold for auto-approval
    
    # UI Settings
    PAGE_TITLE: str = "Enhanced LangGraph Chatbot (Gemini)"
    PAGE_ICON: str = "🤖"
    MAX_CONVERSATION_DISPLAY: int = 10
    
    # Tool Settings
    ENABLE_WEB_SEARCH: bool = True
    ENABLE_CALCULATOR: bool = True
    ENABLE_WEATHER: bool = True
    ENABLE_RAG: bool = True
    
    # Conversation Settings
    DEFAULT_MODE: str = "normal"  # Options: normal, rag, tool_use, advanced
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Check API keys
        validation_results['gemini_key'] = bool(cls.GEMINI_API_KEY)
        validation_results['weather_key'] = bool(cls.WEATHER_API_KEY)
        
        # Check database paths
        validation_results['db_writable'] = os.access(os.path.dirname(cls.DATABASE_PATH) or '.', os.W_OK)
        validation_results['vector_db_accessible'] = True  # Will be created if doesn't exist
        
        # Check model settings
        validation_results['valid_temperature'] = 0.0 <= cls.LLM_TEMPERATURE <= 2.0
        validation_results['valid_max_tokens'] = cls.MAX_TOKENS > 0
        
        return validation_results
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for the chatbot"""
        return f"""You are an advanced AI assistant powered by Google Gemini.

Your capabilities include:
1. General conversation and assistance
2. Web search for current information ({"enabled" if cls.ENABLE_WEB_SEARCH else "disabled"})
3. Weather information lookup ({"enabled" if cls.ENABLE_WEATHER else "disabled"})
4. Mathematical calculations ({"enabled" if cls.ENABLE_CALCULATOR else "disabled"})
5. Knowledge base search using RAG ({"enabled" if cls.ENABLE_RAG else "disabled"})

Guidelines:
- Use tools when appropriate to provide accurate, up-to-date information
- For factual questions, prefer searching the knowledge base first, then web search if needed
- Be helpful, accurate, and conversational
- If you're unsure about something sensitive or important, ask for human approval
- Always explain your reasoning when using tools
- Keep responses concise but informative

Current mode: {cls.DEFAULT_MODE}
"""

# Environment setup helper
def setup_environment():
    """Setup environment and validate configuration"""
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Validate configuration
    validation = ChatbotConfig.validate_config()
    
    print("🔧 Configuration Validation:")
    print("=" * 50)
    
    for key, value in validation.items():
        status = "✅" if value else "❌"
        print(f"{status} {key.replace('_', ' ').title()}: {value}")
    
    print("\\n📋 Setup Instructions:")
    print("=" * 50)
    
    if not validation['gemini_key']:
        print("❌ Gemini API key not found!")
        print("   → Add GEMINI_API_KEY to your .env file")
        print("   → Get your API key from: https://makersuite.google.com/app/apikey")
    
    if not validation['weather_key']:
        print("⚠️  Weather API key not found (optional)")
        print("   → Add WEATHER_API_KEY to your .env file for weather features")
        print("   → Get a free key from: https://openweathermap.org/api")
    
    if not validation['db_writable']:
        print("❌ Database directory not writable!")
        print("   → Check permissions for the database directory")
    
    print("\\n🚀 Ready to start? Run:")
    print("   streamlit run enhanced_frontend.py")
    
    return all(validation[key] for key in ['gemini_key', 'db_writable', 'valid_temperature', 'valid_max_tokens'])

# Advanced features configuration
class AdvancedFeatures:
    """Configuration for advanced features"""
    
    # Conversation Analysis
    ENABLE_SENTIMENT_ANALYSIS: bool = True
    ENABLE_TOPIC_MODELING: bool = False  # Requires additional dependencies
    ENABLE_CONVERSATION_SUMMARIZATION: bool = True
    
    # Security and Safety
    ENABLE_CONTENT_FILTERING: bool = True
    ENABLE_PII_DETECTION: bool = False  # Requires additional dependencies
    ENABLE_TOXICITY_DETECTION: bool = False  # Requires additional dependencies
    
    # Performance Monitoring
    ENABLE_RESPONSE_TIME_TRACKING: bool = True
    ENABLE_TOKEN_USAGE_TRACKING: bool = True
    ENABLE_ERROR_LOGGING: bool = True
    
    # Custom Tools
    CUSTOM_TOOLS_ENABLED: bool = False
    CUSTOM_TOOLS_PATH: str = "./custom_tools"
    
    # Experimental Features
    ENABLE_VOICE_INPUT: bool = False  # Requires speech recognition
    ENABLE_IMAGE_ANALYSIS: bool = False  # Requires vision models
    ENABLE_CODE_EXECUTION: bool = False  # Requires secure sandbox

# Usage example and testing
if __name__ == "__main__":
    print("🤖 Enhanced Chatbot Configuration (Gemini)")
    print("=" * 50)
    
    # Setup and validate environment
    is_ready = setup_environment()
    
    if is_ready:
        print("\\n✅ Configuration is valid! You're ready to go!")
    else:
        print("\\n❌ Please fix the configuration issues above before proceeding.")
    
    # Display current configuration
    print(f"\\n📊 Current Configuration:")
    print(f"   Model: {ChatbotConfig.LLM_MODEL}")
    print(f"   Temperature: {ChatbotConfig.LLM_TEMPERATURE}")
    print(f"   Max Tokens: {ChatbotConfig.MAX_TOKENS}")
    print(f"   Database: {ChatbotConfig.DATABASE_PATH}")
    print(f"   Vector DB: {ChatbotConfig.VECTOR_DB_PATH}")
    print(f"   Default Mode: {ChatbotConfig.DEFAULT_MODE}")