import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import pickle
import json
from pathlib import Path

class DocumentProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document processor with a sentence transformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.documents = {}
        self.embeddings = None
        self.chunks = []
        
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                # Keep overlap
                overlap_sentences = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks
    
    def load_documents_from_directory(self, directory_path: str) -> Dict[str, str]:
        """
        Load all text documents from a directory.
        """
        documents = {}
        directory = Path(directory_path)
        
        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents[file_path.name] = content
                    print(f"Loaded document: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.documents = documents
        return documents
    
    def process_documents(self, chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """
        Process documents into chunks and create embeddings.
        """
        all_chunks = []
        
        for doc_name, content in self.documents.items():
            chunks = self.chunk_text(content, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    'doc_name': doc_name,
                    'chunk_id': f"{doc_name}_chunk_{i}",
                    'content': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_info)
        
        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} chunks from {len(self.documents)} documents")
        return all_chunks
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all document chunks.
        """
        if not self.chunks:
            raise ValueError("No chunks available. Process documents first.")
        
        contents = [chunk['content'] for chunk in self.chunks]
        print(f"Generating embeddings for {len(contents)} chunks...")
        
        embeddings = self.model.encode(contents, show_progress_bar=True)
        self.embeddings = embeddings
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_processed_data(self, output_dir: str):
        """
        Save processed documents, chunks, and embeddings to disk.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save chunks
        with open(output_path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        # Save embeddings
        with open(output_path / "embeddings.npy", 'wb') as f:
            np.save(f, self.embeddings)
        
        # Save document metadata
        metadata = {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'embedding_shape': self.embeddings.shape if self.embeddings is not None else None,
            'model_name': 'all-MiniLM-L6-v2'
        }
        
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved processed data to {output_dir}")
    
    def load_processed_data(self, input_dir: str):
        """
        Load previously processed data from disk.
        """
        input_path = Path(input_dir)
        
        # Load chunks
        with open(input_path / "chunks.json", 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load embeddings
        with open(input_path / "embeddings.npy", 'rb') as f:
            self.embeddings = np.load(f)
        
        # Load metadata
        with open(input_path / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"Loaded processed data: {metadata['num_chunks']} chunks, embedding shape: {self.embeddings.shape}")
    
    def get_document_stats(self) -> Dict:
        """
        Get statistics about loaded documents.
        """
        stats = {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'embedding_shape': self.embeddings.shape if self.embeddings is not None else None,
            'documents': {}
        }
        
        for doc_name, content in self.documents.items():
            word_count = len(content.split())
            char_count = len(content)
            doc_chunks = [chunk for chunk in self.chunks if chunk['doc_name'] == doc_name]
            
            stats['documents'][doc_name] = {
                'word_count': word_count,
                'char_count': char_count,
                'num_chunks': len(doc_chunks)
            }
        
        return stats

def main():
    """
    Main function to process documents and generate embeddings.
    """
    # Initialize processor
    processor = DocumentProcessor()
    
    # Load documents
    docs_dir = "documents"
    processor.load_documents_from_directory(docs_dir)
    
    # Process documents into chunks
    processor.process_documents(chunk_size=512, overlap=50)
    
    # Generate embeddings
    processor.generate_embeddings()
    
    # Save processed data
    processor.save_processed_data("processed_data")
    
    # Print statistics
    stats = processor.get_document_stats()
    print("\nDocument Statistics:")
    print(f"Total documents: {stats['num_documents']}")
    print(f"Total chunks: {stats['num_chunks']}")
    print(f"Embedding shape: {stats['embedding_shape']}")
    
    print("\nDocument details:")
    for doc_name, doc_stats in stats['documents'].items():
        print(f"  {doc_name}: {doc_stats['word_count']} words, {doc_stats['num_chunks']} chunks")

if __name__ == "__main__":
    main()