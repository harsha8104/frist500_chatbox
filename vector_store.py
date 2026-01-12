import numpy as np
import faiss
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle

class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 384, index_type: str = "flat"):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        # Create the appropriate index type
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            chunks: List of document chunks with metadata
            embeddings: Numpy array of embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.normalize_embeddings(embeddings)
        
        # Add to FAISS index
        self.index.add(normalized_embeddings.astype(np.float32))
        
        # Store chunks and embeddings
        self.chunks = chunks
        self.embeddings = normalized_embeddings
        
        print(f"Added {len(chunks)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with scores and metadata
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = self.normalize_embeddings(query_embedding.reshape(1, -1))
        
        # Search the index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                result = {
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'rank': i + 1,
                    'index': int(idx)
                }
                results.append(result)
        
        return results
    
    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Dict]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Multiple query embeddings
            top_k: Number of top results per query
            
        Returns:
            List of search results for each query
        """
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_embeddings))]
        
        # Normalize query embeddings
        query_embeddings = self.normalize_embeddings(query_embeddings)
        
        # Search the index
        scores, indices = self.index.search(query_embeddings.astype(np.float32), top_k)
        
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            query_results = []
            for i, (score, idx) in enumerate(zip(query_scores, query_indices)):
                if idx < len(self.chunks):
                    result = {
                        'chunk': self.chunks[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'index': int(idx)
                    }
                    query_results.append(result)
            all_results.append(query_results)
        
        return all_results
    
    def save(self, directory: str):
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(directory / "faiss.index"))
        
        # Save chunks
        with open(directory / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'num_vectors': self.index.ntotal,
            'num_chunks': len(self.chunks)
        }
        
        with open(directory / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Vector store saved to {directory}")
    
    def load(self, directory: str):
        """
        Load the vector store from disk.
        
        Args:
            directory: Directory containing the saved vector store
        """
        directory = Path(directory)
        
        # Load FAISS index
        self.index = faiss.read_index(str(directory / "faiss.index"))
        
        # Load chunks
        with open(directory / "chunks.json", 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load embeddings
        embeddings_path = directory / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        # Load metadata
        with open(directory / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
        
        print(f"Vector store loaded from {directory}")
        print(f"Contains {self.index.ntotal} vectors from {len(self.chunks)} chunks")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        stats = {
            'num_vectors': self.index.ntotal,
            'num_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'documents': {}
        }
        
        # Group chunks by document
        doc_chunks = {}
        for chunk in self.chunks:
            doc_name = chunk['doc_name']
            if doc_name not in doc_chunks:
                doc_chunks[doc_name] = 0
            doc_chunks[doc_name] += 1
        
        stats['documents'] = doc_chunks
        return stats
    
    def rebuild_index(self):
        """
        Rebuild the FAISS index from stored embeddings.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available to rebuild index")
        
        # Create new index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(100, len(self.chunks))  # Adjust based on data size
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype(np.float32))
        
        print(f"Rebuilt index with {self.index.ntotal} vectors")

# Integration with DocumentProcessor
class RAGSystem:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', vector_store_dir: str = "vector_store"):
        """
        Initialize the complete RAG system.
        """
        from document_processor import DocumentProcessor
        
        self.document_processor = DocumentProcessor(embedding_model_name)
        self.vector_store = FAISSVectorStore()
        self.vector_store_dir = vector_store_dir
    
    def initialize_from_documents(self, docs_directory: str):
        """
        Initialize the RAG system from a directory of documents.
        """
        # Load and process documents
        self.document_processor.load_documents_from_directory(docs_directory)
        chunks = self.document_processor.process_documents()
        embeddings = self.document_processor.generate_embeddings()
        
        # Build vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        # Save for later use
        self.vector_store.save(self.vector_store_dir)
        
        return self.vector_store
    
    def load_existing_vector_store(self):
        """
        Load an existing vector store.
        """
        self.vector_store.load(self.vector_store_dir)
        return self.vector_store
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents given a query.
        """
        # Generate query embedding
        query_embedding = self.document_processor.model.encode([query])
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        return results

def main():
    """
    Example usage of the RAG system.
    """
    # Initialize RAG system
    rag = RAGSystem()
    
    # Initialize from documents (only needed once)
    try:
        rag.initialize_from_documents("documents")
    except Exception as e:
        print(f"Error initializing from documents: {e}")
        print("Attempting to load existing vector store...")
        rag.load_existing_vector_store()
    
    # Test search
    test_queries = [
        "What is the remote work policy?",
        "How much storage do I get with CloudSync Pro?",
        "What are the password requirements?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.search(query, top_k=3)
        
        for i, result in enumerate(results):
            chunk = result['chunk']
            print(f"  Result {i+1} (score: {result['score']:.3f}): {chunk['content'][:200]}...")

if __name__ == "__main__":
    main()