#!/usr/bin/env python3
"""
Simple Demo Version - Bypasses dependency issues
This creates a basic working version without complex dependencies
"""

import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional

# Simple document store
class SimpleDocumentStore:
    def __init__(self):
        self.documents = {}
        self.load_documents()
    
    def load_documents(self):
        """Load documents from the documents directory"""
        docs_dir = "documents"
        if not os.path.exists(docs_dir):
            print(f"Documents directory '{docs_dir}' not found")
            return
        
        txt_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} documents")
        
        for filename in txt_files:
            filepath = os.path.join(docs_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents[filename] = {
                        'content': content,
                        'title': filename.replace('.txt', '').replace('_', ' ').title(),
                        'size': len(content)
                    }
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based search"""
        query_lower = query.lower()
        results = []
        
        for filename, doc in self.documents.items():
            content_lower = doc['content'].lower()
            
            # Simple scoring based on keyword matches
            score = 0
            query_words = query_lower.split()
            
            for word in query_words:
                if word in content_lower:
                    score += content_lower.count(word)
            
            if score > 0:
                # Extract relevant snippet
                snippet = self._extract_snippet(content_lower, query_lower, doc['content'])
                
                results.append({
                    'filename': filename,
                    'title': doc['title'],
                    'score': score,
                    'snippet': snippet,
                    'size': doc['size']
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _extract_snippet(self, content_lower: str, query_lower: str, original_content: str) -> str:
        """Extract relevant snippet around query terms"""
        query_words = query_lower.split()
        
        # Find first occurrence of any query word
        first_pos = len(content_lower)
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1 and pos < first_pos:
                first_pos = pos
        
        if first_pos == len(content_lower):
            return original_content[:200] + "..."
        
        # Extract context around the match
        start = max(0, first_pos - 100)
        end = min(len(original_content), first_pos + 200)
        
        snippet = original_content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(original_content):
            snippet = snippet + "..."
        
        return snippet

class SimpleAIAgent:
    def __init__(self):
        self.document_store = SimpleDocumentStore()
        self.sessions = {}
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Process a user query"""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{random.randint(1000, 9999)}"
        
        # Search documents
        search_results = self.document_store.search(query, top_k=3)
        
        # Generate simple response
        if search_results:
            answer = self._generate_response(query, search_results)
            sources = [result['filename'] for result in search_results]
            confidence = min(0.9, 0.5 + len(search_results) * 0.1)
        else:
            answer = "I couldn't find relevant information in the available documents. Please try rephrasing your question or ask about topics related to company policies, products, or technical documentation."
            sources = []
            confidence = 0.1
        
        # Store session info
        self.sessions[session_id] = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results_count': len(search_results)
        }
        
        return {
            'answer': answer,
            'sources': sources,
            'session_id': session_id,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate a response based on search results"""
        
        # Simple response generation
        response_parts = []
        
        # Add greeting based on time
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning!"
        elif hour < 17:
            greeting = "Good afternoon!"
        else:
            greeting = "Good evening!"
        
        response_parts.append(f"{greeting} Based on the available documents, here's what I found:")
        
        # Add information from search results
        for i, result in enumerate(search_results, 1):
            response_parts.append(f"\n{i}. From {result['title']}:")
            response_parts.append(f"{result['snippet']}")
        
        # Add helpful closing
        response_parts.append("\nI hope this information is helpful! If you need more specific details or have other questions, feel free to ask.")
        
        return " ".join(response_parts)

def main():
    """Main demo function"""
    print("🚀 AI Agent with RAG System - Simple Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = SimpleAIAgent()
    
    print(f"Loaded {len(agent.document_store.documents)} documents")
    print("Available documents:")
    for filename, doc in agent.document_store.documents.items():
        print(f"  - {doc['title']}")
    
    print("\n" + "=" * 50)
    print("🤖 Ready to answer your questions!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        query = input("\n❓ Ask a question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            response = agent.process_query(query)
            
            print(f"\n🤖 Answer: {response['answer']}")
            
            if response['sources']:
                print(f"📚 Sources: {', '.join(response['sources'])}")
            
            print(f"🔍 Confidence: {response['confidence']:.1%}")
            print(f"📝 Session ID: {response['session_id']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()