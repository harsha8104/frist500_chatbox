ru#!/usr/bin/env python3
"""
Script to initialize the AI Agent system by processing documents and building the vector store.
Run this script once to set up the document processing pipeline.
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Dict
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    try:
        # Skip detailed dependency check for now
        logger.info("Dependencies check skipped - proceeding with initialization")
        return True
    except Exception as e:
        logger.error(f"Dependency check error: {e}")
        return False

def process_documents():
    """Process documents and create vector store."""
    try:
        from document_processor import DocumentProcessor
        from vector_store import FAISSVectorStore
        
        # Initialize document processor
        logger.info("Initializing document processor...")
        processor = DocumentProcessor()
        
        # Check if documents directory exists
        docs_dir = Path("documents")
        if not docs_dir.exists():
            logger.error(f"Documents directory '{docs_dir}' not found")
            logger.error("Please create the documents directory and add your text files")
            return False
        
        # Count documents
        txt_files = list(docs_dir.glob("*.txt"))
        if not txt_files:
            logger.error("No .txt files found in documents directory")
            return False
        
        logger.info(f"Found {len(txt_files)} text files to process")
        
        # Load documents
        logger.info("Loading documents...")
        processor.load_documents_from_directory(str(docs_dir))
        
        # Process documents into chunks
        logger.info("Processing documents into chunks...")
        chunks = processor.process_documents(chunk_size=512, overlap=50)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = processor.generate_embeddings()
        
        # Create vector store
        logger.info("Creating FAISS vector store...")
        vector_store = FAISSVectorStore()
        vector_store.add_documents(chunks, embeddings)
        
        # Save vector store
        vector_store_dir = Path("vector_store")
        vector_store_dir.mkdir(exist_ok=True)
        vector_store.save(str(vector_store_dir))
        
        # Save processed data for backup
        processed_data_dir = Path("processed_data")
        processed_data_dir.mkdir(exist_ok=True)
        processor.save_processed_data(str(processed_data_dir))
        
        # Print statistics
        stats = vector_store.get_stats()
        logger.info("Document processing completed successfully!")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        return False

def test_rag_system():
    """Test the RAG system with sample queries."""
    try:
        from vector_store import RAGSystem
        
        logger.info("Testing RAG system...")
        
        # Initialize RAG system
        rag = RAGSystem()
        rag.load_existing_vector_store()
        
        # Test queries
        test_queries = [
            "What is the remote work policy?",
            "How much storage do I get with CloudSync Pro?",
            "What are the password requirements?",
            "Tell me about employee benefits",
            "What is the API documentation about?"
        ]
        
        logger.info("Running test queries...")
        for query in test_queries:
            logger.info(f"\\nQuery: {query}")
            results = rag.search(query, top_k=3)
            
            if results:
                logger.info(f"Found {len(results)} relevant results")
                for i, result in enumerate(results[:2]):  # Show top 2 results
                    chunk = result['chunk']
                    score = result['score']
                    doc_name = chunk['doc_name']
                    content_preview = chunk['content'][:150].replace('\\n', ' ')
                    logger.info(f"  Result {i+1} (score: {score:.3f}, doc: {doc_name}): {content_preview}...")
            else:
                logger.info("  No relevant results found")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing RAG system: {e}")
        return False

def create_sample_documents():
    """Create sample documents if none exist."""
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Check if documents already exist
    existing_docs = list(docs_dir.glob("*.txt"))
    if existing_docs:
        logger.info(f"Found {len(existing_docs)} existing documents")
        return True
    
    logger.info("Creating sample documents...")
    
    # Sample document content
    sample_docs = {
        "welcome.txt": """Welcome to our company!

We are excited to have you join our team. This document provides an overview of our company culture, values, and basic information for new employees.

Our mission is to deliver exceptional products and services while maintaining a positive work environment that fosters growth and innovation.

Key values:
- Customer first
- Innovation
- Teamwork
- Integrity
- Excellence

If you have any questions, please don't hesitate to reach out to your manager or HR representative.""",
        
        "quick_reference.txt": """Quick Reference Guide

Emergency Contacts:
- IT Support: ext. 1234 or itsupport@company.com
- HR Department: ext. 5678 or hr@company.com
- Security: ext. 911 or security@company.com

Common Systems:
- Email: mail.company.com
- Intranet: intranet.company.com
- Time Tracking: time.company.com
- Benefits Portal: benefits.company.com

Office Hours: 9:00 AM - 5:00 PM, Monday through Friday
Lunch Break: 12:00 PM - 1:00 PM (flexible)

For more detailed information, please refer to the employee handbook and other policy documents."""
    }
    
    for filename, content in sample_docs.items():
        file_path = docs_dir / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created sample document: {filename}")
    
    return True

def main():
    """Main function to initialize the system."""
    logger.info("Starting AI Agent system initialization...")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return 1
    
    # Create sample documents if needed
    if not create_sample_documents():
        logger.error("Failed to create sample documents")
        return 1
    
    # Process documents
    if not process_documents():
        logger.error("Document processing failed")
        return 1
    
    # Test RAG system
    if not test_rag_system():
        logger.error("RAG system test failed")
        return 1
    
    logger.info("\\nAI Agent system initialization completed successfully!")
    logger.info("You can now run the FastAPI server with: python main.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())