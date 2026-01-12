from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import logging
from datetime import datetime
import uuid
import socket
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def find_free_port(start_port=8001):
    """Find the next available port starting from start_port."""
    port = start_port
    while port < 65536:  # Max port number
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
    raise OSError("No free ports available")

# Import our modules
from ai_agent import AIAgent, AgentResponse
from vector_store import RAGSystem
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="AI-powered question answering system with RAG capabilities",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")

class AskResponse(BaseModel):
    answer: str = Field(..., description="The AI agent's response")
    sources: List[str] = Field(default_factory=list, description="List of source documents used")
    confidence: float = Field(..., description="Confidence score (0-1)")
    session_id: str = Field(..., description="Session ID for this conversation")
    timestamp: str = Field(..., description="Timestamp of the response")
    query_type: str = Field(..., description="Type of query classified by the agent")

class SessionInfoResponse(BaseModel):
    session_id: str
    total_interactions: int
    query_types: Dict[str, int]
    session_start: str
    last_interaction: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent and related services on startup."""
    global agent
    
    try:
        logger.info("Initializing AI Agent...")
        
        # Check for required environment variables
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY", 
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.warning("Using mock responses for development")
            
            # Create a mock agent for development
            class MockAgent:
                def process_query(self, query: str, session_id: Optional[str] = None):
                    return AgentResponse(
                        answer=f"This is a mock response to: {query[:50]}...",
                        sources=["mock_document.txt"],
                        confidence=0.8,
                        query_type="general"
                    )
            
            agent = MockAgent()
        else:
            # Initialize real agent
            agent = AIAgent(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            )
            
            # Initialize RAG system if documents exist
            if os.path.exists("documents") and os.path.exists("vector_store"):
                logger.info("Loading existing vector store...")
                try:
                    agent.rag_system.load_existing_vector_store()
                    logger.info("Vector store loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load vector store: {e}")
            
        logger.info("AI Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {e}")
        raise

@app.get("/")
async def root():
    """Serve the web interface."""
    return FileResponse("web_interface.html", media_type="text/html")

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "AI Agent API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services_status = {
        "api": "healthy",
        "agent": "healthy" if agent else "unhealthy",
        "rag_system": "healthy" if hasattr(agent, 'rag_system') else "not_configured"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        services=services_status
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Main endpoint for asking questions to the AI agent.
    
    This endpoint accepts user queries and returns AI-generated responses
    with optional source documents and session management.
    """
    try:
        logger.info(f"Received query: {request.query[:50]}...")
        
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = str(uuid.uuid4())
        
        # Process the query
        agent_response = agent.process_query(
            query=request.query,
            session_id=request.session_id
        )
        
        # Create response
        response = AskResponse(
            answer=agent_response.answer,
            sources=agent_response.sources,
            confidence=agent_response.confidence,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat(),
            query_type=agent_response.query_type.value
        )
        
        logger.info(f"Query processed successfully. Confidence: {agent_response.confidence:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """
    Get information about a specific session.
    """
    try:
        if not hasattr(agent, 'get_session_info'):
            raise HTTPException(status_code=501, detail="Session management not available")
        
        session_info = agent.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionInfoResponse(
            session_id=session_id,
            total_interactions=session_info['total_interactions'],
            query_types=session_info['query_types'],
            session_start=session_info['session_start'],
            last_interaction=session_info['last_interaction']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session info: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a session's conversation history.
    """
    try:
        if not hasattr(agent, 'clear_session'):
            raise HTTPException(status_code=501, detail="Session management not available")
        
        agent.clear_session(session_id)
        
        return {"message": f"Session {session_id} cleared successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.post("/process-documents")
async def process_documents():
    """
    Endpoint to trigger document processing and embedding generation.
    This should be called when new documents are added or updated.
    """
    try:
        logger.info("Starting document processing...")
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Load documents
        docs_dir = "documents"
        if not os.path.exists(docs_dir):
            raise HTTPException(status_code=404, detail="Documents directory not found")
        
        doc_processor.load_documents_from_directory(docs_dir)
        
        # Process documents
        chunks = doc_processor.process_documents()
        embeddings = doc_processor.generate_embeddings()
        
        # Build vector store
        from vector_store import FAISSVectorStore
        vector_store = FAISSVectorStore()
        vector_store.add_documents(chunks, embeddings)
        
        # Save vector store
        vector_store.save("vector_store")
        
        # Update agent's RAG system
        if hasattr(agent, 'rag_system'):
            agent.rag_system.load_existing_vector_store()
        
        stats = vector_store.get_stats()
        
        logger.info(f"Document processing completed. Stats: {stats}")
        
        return {
            "message": "Documents processed successfully",
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.get("/document-stats")
async def get_document_stats():
    """
    Get statistics about processed documents and vector store.
    """
    try:
        if not os.path.exists("vector_store"):
            return {"message": "No documents have been processed yet"}
        
        from vector_store import FAISSVectorStore
        vector_store = FAISSVectorStore()
        vector_store.load("vector_store")
        
        stats = vector_store.get_stats()
        
        return {
            "message": "Document statistics retrieved successfully",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An internal server error occurred",
                "status_code": 500,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

if __name__ == "__main__":
    import uvicorn

    # Run the server
    default_port = int(os.getenv("PORT", 8001))
    port = find_free_port(default_port)
    host = os.getenv("HOST", "127.0.0.1")  # Changed from 0.0.0.0 to localhost for local access

    print(f"Starting server on {host}:{port}")
    if port != default_port:
        print(f"Port {default_port} was busy, using port {port} instead")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True if os.getenv("ENVIRONMENT") == "development" else False
    )
