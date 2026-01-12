# AI Agent with RAG System

A sophisticated AI agent built with FastAPI that combines Azure OpenAI with Retrieval-Augmented Generation (RAG) to answer questions about company documents. The system features intelligent query classification, session-based memory, and comprehensive document retrieval capabilities.

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI       │────▶│   AI Agent      │
│   (User Query)  │     │   Backend       │     │   (Logic)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                       ┌─────────────────┐              │
                       │   FAISS Vector  │◀─────────────┤
                       │   Store         │              │
                       └─────────────────┘              │
                                                        │
                       ┌─────────────────┐              │
                       │   Azure OpenAI  │◀─────────────┤
                       │   (LLM)         │              │
                       └─────────────────┘              │
                                                        │
                       ┌─────────────────┐              │
                       │   Document      │─────────────▶│
                       │   Processing    │              │
                       └─────────────────┘              │
                                                        │
                       ┌─────────────────┐              │
                       │   Session       │◀─────────────┤
                       │   Memory        │              │
                       └─────────────────┘
```

## 🚀 Features

- **AI Agent**: Intelligent query classification and response generation
- **RAG System**: Document retrieval with FAISS vector store
- **Session Memory**: Maintains conversation context across queries
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **Azure OpenAI Integration**: Leverages Azure's enterprise-grade AI services
- **Docker Support**: Containerized deployment ready
- **Comprehensive Logging**: Detailed request/response tracking

## 🛠️ Tech Stack

- **Backend**: Python 3.9+, FastAPI
- **AI/ML**: Azure OpenAI, Sentence Transformers, FAISS
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence-BERT for document embeddings
- **Session Management**: In-memory session store
- **Deployment**: Docker, Azure App Service
- **Documentation**: Auto-generated OpenAPI/Swagger docs

## 📋 Prerequisites

- Python 3.9 or higher
- Azure OpenAI service subscription
- Docker (optional, for containerized deployment)
- 8GB+ RAM recommended for vector operations

## 🔧 Local Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd ai-agent-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Azure OpenAI Configuration

1. **Create Azure OpenAI Resource**:
   - Go to Azure Portal → Create Resource → Azure OpenAI
   - Select your subscription and resource group
   - Choose region (e.g., East US)
   - Name your resource (e.g., `ai-agent-openai`)

2. **Deploy Model**:
   - Go to Azure OpenAI Studio
   - Deploy a model (e.g., `gpt-35-turbo` or `gpt-4`)
   - Note the deployment name

3. **Get API Keys**:
   - Go to Azure OpenAI resource → Keys and Endpoint
   - Copy the endpoint URL and one of the keys

4. **Configure Environment**:
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your Azure OpenAI details
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2023-05-15
```

### 3. Initialize the System

```bash
# Process documents and build vector store
python initialize.py

# This will:
# - Process all documents in the `documents/` directory
# - Generate embeddings using Sentence Transformers
# - Build FAISS vector index
# - Test the RAG system with sample queries
```

### 4. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Endpoint**: http://localhost:8000/ask

## 📖 API Usage

### POST /ask - Query the AI Agent

**Request:**
```json
{
  "query": "What is the remote work policy?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "answer": "Based on our company policy, employees may work remotely up to 3 days per week...",
  "sources": ["company_remote_work_policy.txt", "employee_handbook.txt"],
  "session_id": "session-12345",
  "confidence": 0.85,
  "query_type": "document_search"
}
```

### GET /health - Health Check

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "vector_store_loaded": true,
  "azure_openai_connected": true
}
```

## 📁 Project Structure

```
ai-agent-rag-system/
├── 📄 main.py                    # FastAPI application
├── 📄 ai_agent.py              # AI Agent with Azure OpenAI
├── 📄 document_processor.py     # Document processing and embeddings
├── 📄 vector_store.py          # FAISS vector store implementation
├── 📄 initialize.py            # System initialization script
├── 📄 requirements.txt         # Python dependencies
├── 📄 Dockerfile               # Docker configuration
├── 📄 docker-compose.yml       # Docker Compose setup
├── 📄 .env.example             # Environment variables template
├── 📄 documents/               # Document repository
│   ├── company_remote_work_policy.txt
│   ├── product_faq_cloudsync.txt
│   ├── technical_api_guide.txt
│   ├── employee_benefits_handbook.txt
│   └── it_security_policy.txt
└── 📄 README.md               # This file
```

## 🐳 Docker Deployment

### Local Docker Deployment

```bash
# Build and run with Docker
docker build -t ai-agent-rag .
docker run -p 8000:8000 --env-file .env ai-agent-rag

# Or use Docker Compose
docker-compose up -d
```

### Production Docker Deployment

```bash
# Production setup with Nginx reverse proxy
docker-compose --profile production up -d
```

## ☁️ Azure Deployment

### Option 1: Azure App Service

1. **Create App Service Plan**:
```bash
# Create resource group
az group create --name ai-agent-rg --location eastus

# Create app service plan
az appservice plan create --name ai-agent-plan --resource-group ai-agent-rg --sku B1 --is-linux
```

2. **Create Web App**:
```bash
# Create web app
az webapp create --resource-group ai-agent-rg --plan ai-agent-plan --name ai-agent-app --deployment-container-image-name ai-agent-rag:latest
```

3. **Configure Environment Variables**:
```bash
# Set environment variables
az webapp config appsettings set --resource-group ai-agent-rg --name ai-agent-app --settings AZURE_OPENAI_ENDPOINT="your-endpoint" AZURE_OPENAI_API_KEY="your-key"
```

### Option 2: Azure Container Instances

```bash
# Create container instance
az container create --resource-group ai-agent-rg --name ai-agent-container --image ai-agent-rag:latest --ports 8000 --environment-variables AZURE_OPENAI_ENDPOINT="your-endpoint" AZURE_OPENAI_API_KEY="your-key"
```

### Option 3: Azure Functions (Serverless)

For serverless deployment, modify the application to use Azure Functions with HTTP triggers.

## 🔍 Design Decisions

### 1. **FAISS vs Azure AI Search**
- **Choice**: FAISS for vector storage
- **Reason**: Cost-effective, no additional service dependencies, sufficient for document retrieval
- **Alternative**: Azure AI Search for enterprise-scale deployments

### 2. **Session-Based Memory**
- **Choice**: In-memory session store
- **Reason**: Simple, fast, suitable for single-instance deployments
- **Limitation**: Not persistent across restarts or multiple instances
- **Future**: Redis or Azure Cache for distributed scenarios

### 3. **Sentence Transformers**
- **Choice**: `all-MiniLM-L6-v2` model
- **Reason**: Good balance between performance and accuracy, 384-dimensional embeddings
- **Alternative**: Larger models for better accuracy or Azure OpenAI embeddings

### 4. **Document Chunking Strategy**
- **Choice**: 512 tokens with 50 token overlap
- **Reason**: Balances context preservation with retrieval accuracy
- **Alternative**: Adaptive chunking based on document structure

## ⚠️ Limitations & Future Improvements

### Current Limitations

1. **Memory Persistence**: Session memory is lost on server restart
2. **Single Instance**: Not designed for multi-instance deployments
3. **Document Format**: Only supports text files currently
4. **Real-time Updates**: No live document updates without reprocessing
5. **Rate Limiting**: Basic rate limiting, could be more sophisticated

### Future Enhancements

1. **Distributed Memory**:
   - Implement Redis for session persistence
   - Support for user authentication and personalized memory

2. **Advanced RAG Features**:
   - Hybrid search (semantic + keyword)
   - Multi-modal document support (PDF, images)
   - Real-time document indexing

3. **Enhanced Monitoring**:
   - Azure Application Insights integration
   - Performance metrics and alerting
   - Usage analytics dashboard

4. **Security Improvements**:
   - API key authentication
   - Request validation and sanitization
   - Audit logging for compliance

5. **Scalability**:
   - Kubernetes deployment support
   - Auto-scaling based on load
   - Multi-region deployment

6. **Advanced AI Features**:
   - Multi-language support
   - Fine-tuned models for specific domains
   - Integration with other Azure AI services

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests (when implemented)
python -m pytest tests/
```

### Integration Tests
```bash
# Test API endpoints
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

### Load Testing
```bash
# Use locust for load testing
pip install locust
locust -f load_test.py --host http://localhost:8000
```

## 📊 Performance Considerations

- **Vector Search**: FAISS provides sub-second search for thousands of documents
- **Memory Usage**: ~1GB RAM for 10,000 document chunks
- **Response Time**: 2-5 seconds for complex queries including document retrieval
- **Concurrent Users**: Supports 50+ concurrent users on standard hardware

## 🔧 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Failed**:
   - Verify endpoint URL and API key
   - Check Azure subscription and resource status
   - Ensure model deployment is active

2. **Vector Store Not Found**:
   - Run `python initialize.py` to build vector store
   - Check file permissions in `vector_store/` directory

3. **Out of Memory Errors**:
   - Reduce document chunk size
   - Limit number of documents processed
   - Increase Docker memory limits

4. **Slow Response Times**:
   - Check Azure OpenAI service health
   - Optimize document chunking strategy
   - Consider upgrading Azure OpenAI tier

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Azure OpenAI service status
3. Examine application logs for specific errors
4. Ensure all environment variables are correctly set

## 📄 License

This project is created for educational purposes. Please ensure compliance with Azure OpenAI usage policies and your organization's data governance requirements.

---

**Built with ❤️ using FastAPI, Azure OpenAI, and FAISS**