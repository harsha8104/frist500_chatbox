import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import AzureOpenAI
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the agent can handle."""
    GENERAL = "general"
    DOCUMENT_SPECIFIC = "document_specific"
    GREETING = "greeting"
    CLARIFICATION = "clarification"

@dataclass
class AgentResponse:
    """Response from the AI agent."""
    answer: str
    sources: List[str]
    confidence: float
    query_type: QueryType
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

class SessionMemory:
    """Simple session-based memory for the AI agent."""
    
    def __init__(self, session_id: str, max_history: int = 10):
        self.session_id = session_id
        self.max_history = max_history
        self.conversation_history = []
        self.context = {}
    
    def add_interaction(self, query: str, response: AgentResponse):
        """Add an interaction to the conversation history."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response.answer,
            "sources": response.sources,
            "query_type": response.query_type.value
        }
        
        self.conversation_history.append(interaction)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_recent_context(self, num_interactions: int = 3) -> str:
        """Get recent conversation context."""
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-num_interactions:]
        context_parts = []
        
        for interaction in recent:
            context_parts.append(f"User: {interaction['query']}")
            context_parts.append(f"Assistant: {interaction['response']}")
        
        return "\\n".join(context_parts)
    
    def get_session_stats(self) -> Dict:
        """Get statistics about the session."""
        if not self.conversation_history:
            return {"total_interactions": 0}
        
        query_types = {}
        for interaction in self.conversation_history:
            qt = interaction['query_type']
            query_types[qt] = query_types.get(qt, 0) + 1
        
        return {
            "total_interactions": len(self.conversation_history),
            "query_types": query_types,
            "session_start": self.conversation_history[0]['timestamp'],
            "last_interaction": self.conversation_history[-1]['timestamp']
        }

class AIAgent:
    """AI Agent with Azure OpenAI integration and tool calling capabilities."""

    def __init__(self,
                 azure_endpoint: str,
                 azure_api_key: str,
                 deployment_name: str,
                 api_version: str = "2023-12-01-preview"):
        """
        Initialize the AI agent with Azure OpenAI.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            deployment_name: Azure OpenAI deployment name
            api_version: API version to use
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.sessions = {}  # Session memory storage

        # Initialize RAG system
        from vector_store import RAGSystem
        self.rag_system = RAGSystem()
        try:
            self.rag_system.load_existing_vector_store()
            logger.info("RAG system loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load RAG system: {e}")
            logger.info("RAG system will be initialized when needed")
    
    def get_or_create_session(self, session_id: str) -> SessionMemory:
        """Get or create a session memory."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id)
        return self.sessions[session_id]
    
    def classify_query(self, query: str) -> QueryType:
        """Classify the type of query."""
        query_lower = query.lower().strip()
        
        # Greeting detection
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in query_lower for greeting in greetings):
            return QueryType.GREETING
        
        # Document-specific keywords
        doc_keywords = ['policy', 'procedure', 'document', 'guide', 'handbook', 'benefits', 
                         'remote work', 'security', 'api', 'cloudsync', 'storage', 'password']
        if any(keyword in query_lower for keyword in doc_keywords):
            return QueryType.DOCUMENT_SPECIFIC
        
        # Clarification requests
        clarification_words = ['what do you mean', 'can you clarify', 'explain more', 'details']
        if any(phrase in query_lower for phrase in clarification_words):
            return QueryType.CLARIFICATION
        
        return QueryType.GENERAL
    
    def needs_document_search(self, query: str, query_type: QueryType) -> bool:
        """Determine if the query needs document search."""
        if query_type == QueryType.DOCUMENT_SPECIFIC:
            return True
        
        # Check for specific patterns that indicate document search needed
        specific_patterns = [
            'what is the policy', 'according to', 'document mentions', 'handbook says',
            'procedure for', 'requirements for', 'guidelines for'
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in specific_patterns)
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents using RAG."""
        try:
            results = self.rag_system.search(query, top_k)
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def format_context_from_documents(self, search_results: List[Dict]) -> str:
        """Format search results into context for the LLM."""
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results):
            chunk = result['chunk']
            score = result['score']
            doc_name = chunk['doc_name']
            content = chunk['content']
            
            context_parts.append(f"Document {i+1} ({doc_name}, relevance: {score:.3f}):\\n{content}")
            sources.append(doc_name)
        
        return "\\n\\n".join(context_parts), list(set(sources))
    
    def create_system_prompt(self, query_type: QueryType, has_context: bool) -> str:
        """Create appropriate system prompt based on query type and context."""
        base_prompt = """You are a helpful AI assistant for a company. You provide accurate, helpful, and professional responses to employee questions."""
        
        if has_context:
            context_instruction = """
Use the provided document context to answer the user's question. Be specific and reference the relevant documents when appropriate. If the context doesn't contain enough information to fully answer the question, say so clearly.
"""
        else:
            context_instruction = """
Answer the user's question based on your general knowledge. If you don't have enough information or context to answer accurately, let the user know and suggest they check the relevant company documents or contact the appropriate department.
"""
        
        if query_type == QueryType.GREETING:
            return base_prompt + """
Respond to greetings in a friendly, professional manner. Keep responses brief but welcoming.""" + context_instruction
        elif query_type == QueryType.DOCUMENT_SPECIFIC:
            return base_prompt + """
Focus on providing accurate information from company documents. Be specific about policies, procedures, and guidelines. If referencing specific documents, mention them by name.""" + context_instruction
        else:
            return base_prompt + context_instruction
    
    def call_openai(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call Azure OpenAI with the given messages and tools."""
        try:
            if tools:
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=messages,
                    tools=tools,
                    temperature=0.7,
                    max_tokens=1000
                )
            else:
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
            
            return response
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            raise
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> AgentResponse:
        """
        Process a user query and return a response.
        
        Args:
            query: User's query
            session_id: Optional session ID for maintaining conversation history
            
        Returns:
            AgentResponse with answer, sources, and metadata
        """
        # Get or create session
        if session_id:
            session = self.get_or_create_session(session_id)
            recent_context = session.get_recent_context()
        else:
            session = None
            recent_context = ""
        
        # Classify query
        query_type = self.classify_query(query)
        logger.info(f"Query classified as: {query_type.value}")
        
        # Determine if document search is needed
        needs_search = self.needs_document_search(query, query_type)
        search_results = []
        context = ""
        sources = []
        
        if needs_search:
            logger.info("Searching documents...")
            search_results = self.search_documents(query, top_k=5)
            if search_results:
                context, sources = self.format_context_from_documents(search_results)
                logger.info(f"Found {len(sources)} relevant documents")
            else:
                logger.warning("No relevant documents found")
        
        # Create system prompt
        system_prompt = self.create_system_prompt(query_type, bool(context))
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent context if available
        if recent_context:
            messages.append({
                "role": "system", 
                "content": f"Recent conversation context:\\n{recent_context}"
            })
        
        # Add document context if available
        if context:
            messages.append({
                "role": "system", 
                "content": f"Relevant document context:\\n{context}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        try:
            # Call OpenAI
            response = self.call_openai(messages)
            answer = response.choices[0].message.content
            
            # Calculate confidence based on search results and response characteristics
            confidence = self.calculate_confidence(search_results, answer)
            
            # Create response
            agent_response = AgentResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                query_type=query_type
            )
            
            # Add to session memory
            if session:
                session.add_interaction(query, agent_response)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return AgentResponse(
                answer="I apologize, but I encountered an error processing your question. Please try again or contact support if the issue persists.",
                sources=[],
                confidence=0.0,
                query_type=query_type
            )
    
    def calculate_confidence(self, search_results: List[Dict], answer: str) -> float:
        """Calculate confidence score based on various factors."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we have good search results
        if search_results:
            # Check top result score
            top_score = search_results[0]['score']
            if top_score > 0.8:
                confidence += 0.3
            elif top_score > 0.6:
                confidence += 0.2
            elif top_score > 0.4:
                confidence += 0.1
        
        # Check if answer mentions specific documents or sources
        if any(word in answer.lower() for word in ['document', 'policy', 'according to', 'section']):
            confidence += 0.1
        
        # Check if answer contains uncertainty indicators
        uncertainty_words = ['not sure', 'might be', 'could be', 'possibly', 'unclear']
        if any(phrase in answer.lower() for phrase in uncertainty_words):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return session.get_session_stats()
    
    def clear_session(self, session_id: str):
        """Clear a session's memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

# Tool definitions for function calling
def get_weather_tool() -> Dict:
    """Get weather information tool."""
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or address"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }

def get_document_summary_tool() -> Dict:
    """Get document summary tool."""
    return {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": "Get a summary of a specific document",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_name": {
                        "type": "string",
                        "description": "Name of the document to summarize"
                    }
                },
                "required": ["document_name"]
            }
        }
    }

if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize agent
    agent = AIAgent(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
    )
    
    # Test queries
    test_queries = [
        "Hello!",
        "What is the remote work policy?",
        "How much vacation time do I get?",
        "What are the password requirements?",
        "Tell me about CloudSync Pro storage limits."
    ]
    
    session_id = "test_session_123"
    
    for query in test_queries:
        print(f"\\nQuery: {query}")
        response = agent.process_query(query, session_id)
        print(f"Answer: {response.answer}")
        print(f"Sources: {response.sources}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Query Type: {response.query_type.value}")