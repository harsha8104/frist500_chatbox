#!/bin/bash
# Azure deployment script for AI Agent with RAG System

set -e

echo "🚀 Starting Azure deployment for AI Agent RAG System..."

# Configuration
RESOURCE_GROUP="ai-agent-rg"
LOCATION="eastus"
APP_SERVICE_PLAN="ai-agent-plan"
WEB_APP="ai-agent-app-$(date +%s)"
CONTAINER_REGISTRY="aiaagentacr$(date +%s)"
OPENAI_RESOURCE="ai-agent-openai-$(date +%s)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is logged in to Azure
if ! az account show &> /dev/null; then
    print_error "Please login to Azure using 'az login'"
    exit 1
fi

print_status "Azure CLI check passed"

# Create resource group
print_status "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
print_status "Creating Azure Container Registry: $CONTAINER_REGISTRY"
az acr create --resource-group $RESOURCE_GROUP --name $CONTAINER_REGISTRY --sku Basic --admin-enabled true

# Build and push Docker image
print_status "Building and pushing Docker image..."
docker build -t ai-agent-rag:latest .
docker tag ai-agent-rag:latest $CONTAINER_REGISTRY.azurecr.io/ai-agent-rag:latest

# Login to ACR
az acr login --name $CONTAINER_REGISTRY

# Push image
docker push $CONTAINER_REGISTRY.azurecr.io/ai-agent-rag:latest

# Create App Service Plan
print_status "Creating App Service Plan: $APP_SERVICE_PLAN"
az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --sku B1 --is-linux

# Create Web App with container
print_status "Creating Web App: $WEB_APP"
az webapp create --resource-group $RESOURCE_GROUP --plan $APP_SERVICE_PLAN --name $WEB_APP --deployment-container-image-name $CONTAINER_REGISTRY.azurecr.io/ai-agent-rag:latest

# Configure container settings
print_status "Configuring container settings..."
az webapp config container set --name $WEB_APP --resource-group $RESOURCE_GROUP --docker-registry-server-url https://$CONTAINER_REGISTRY.azurecr.io --docker-registry-server-user $CONTAINER_REGISTRY --docker-registry-server-password $(az acr credential show --name $CONTAINER_REGISTRY --query passwords[0].value -o tsv)

# Create Azure OpenAI resource
print_status "Creating Azure OpenAI resource: $OPENAI_RESOURCE"
az cognitiveservices account create --name $OPENAI_RESOURCE --resource-group $RESOURCE_GROUP --location $LOCATION --kind OpenAI --sku S0 --custom-domain $OPENAI_RESOURCE

# Deploy GPT model
print_status "Deploying GPT model..."
az cognitiveservices account deployment create --name $OPENAI_RESOURCE --resource-group $RESOURCE_GROUP --deployment-name gpt-35-turbo --model-name gpt-35-turbo --model-version "0613" --model-format OpenAI --sku-name "Standard" --sku-capacity 1

# Get Azure OpenAI credentials
OPENAI_ENDPOINT=$(az cognitiveservices account show --name $OPENAI_RESOURCE --resource-group $RESOURCE_GROUP --query properties.endpoint -o tsv)
OPENAI_KEY=$(az cognitiveservices account keys list --name $OPENAI_RESOURCE --resource-group $RESOURCE_GROUP --query key1 -o tsv)

# Set environment variables for the web app
print_status "Setting environment variables..."
az webapp config appsettings set --name $WEB_APP --resource-group $RESOURCE_GROUP --settings \
    AZURE_OPENAI_ENDPOINT="$OPENAI_ENDPOINT" \
    AZURE_OPENAI_API_KEY="$OPENAI_KEY" \
    AZURE_OPENAI_DEPLOYMENT_NAME="gpt-35-turbo" \
    AZURE_OPENAI_API_VERSION="2023-05-15" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED="1"

# Configure health check
print_status "Configuring health check..."
az webapp config set --name $WEB_APP --resource-group $RESOURCE_GROUP --always-on true

# Scale up if needed (optional)
print_warning "Consider scaling up the App Service Plan for production use"
# az appservice plan update --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --sku P1V2

# Get the web app URL
WEB_APP_URL=$(az webapp show --name $WEB_APP --resource-group $RESOURCE_GROUP --query defaultHostName -o tsv)

print_status "🎉 Deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Web App: $WEB_APP"
echo "  URL: https://$WEB_APP_URL"
echo "  Azure OpenAI Resource: $OPENAI_RESOURCE"
echo ""
echo "🔧 Next Steps:"
echo "  1. Upload documents to initialize the system"
echo "  2. Access the API at: https://$WEB_APP_URL/docs"
echo "  3. Test the /ask endpoint with your queries"
echo ""
echo "📁 To upload documents and initialize the system:"
echo "  1. Add your .txt files to the documents/ directory"
echo "  2. Run: python initialize.py (locally first to test)"
echo "  3. The system will automatically process documents on startup"
echo ""
echo "💡 Tips:"
echo "  - Monitor logs: az webapp log tail --name $WEB_APP --resource-group $RESOURCE_GROUP"
echo "  - Scale up: Consider upgrading to Premium V2 tier for production"
echo "  - Security: Add authentication and CORS configuration as needed"

# Save deployment information
cat > deployment_info.json << EOF
{
  "resource_group": "$RESOURCE_GROUP",
  "web_app": "$WEB_APP",
  "web_app_url": "https://$WEB_APP_URL",
  "container_registry": "$CONTAINER_REGISTRY",
  "openai_resource": "$OPENAI_RESOURCE",
  "location": "$LOCATION",
  "deployment_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

print_status "Deployment information saved to deployment_info.json"

# Optional: Create monitoring and alerts
print_warning "Consider setting up monitoring and alerts for production use"
# az monitor metrics alert create --name "HighResponseTime" --resource-group $RESOURCE_GROUP --scopes "/subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Web/sites/$WEB_APP" --condition "avg requests/duration > 5000" --window-size 5m --evaluation-frequency 1m

echo ""
print_status "🚀 Azure deployment script completed!"