"""
Ajatuskumppani RAG (Retrieval-Augmented Generation) System
Enables AI to access and use external knowledge
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure for RAG"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class AjatusRAGRetriever:
    """RAG retriever for Ajatuskumppani"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        index_path: Optional[str] = None
    ):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.index_path = index_path or "./data/faiss_index"
        
    def load_embedding_model(self):
        """Load sentence transformer model"""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for multiple documents"""
        logger.info(f"Embedding {len(documents)} documents...")
        
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        logger.info("Documents embedded")
        return documents
    
    def build_index(self, documents: List[Document]):
        """Build FAISS index from documents"""
        logger.info("Building FAISS index...")
        
        # Embed documents if not already embedded
        if documents[0].embedding is None:
            documents = self.embed_documents(documents)
        
        # Get embedding dimension
        dim = documents[0].embedding.shape[0]
        
        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index
        embeddings = np.array([doc.embedding for doc in documents])
        self.index.add(embeddings)
        
        # Store documents
        self.documents = documents
        
        logger.info(f"Index built with {len(documents)} documents")
    
    def save_index(self, path: Optional[str] = None):
        """Save FAISS index and documents to disk"""
        path = path or self.index_path
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save documents (without embeddings to save space)
        docs_data = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            for doc in self.documents
        ]
        
        with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: Optional[str] = None):
        """Load FAISS index and documents from disk"""
        path = path or self.index_path
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load documents
        with open(os.path.join(path, "documents.json"), "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        
        self.documents = [
            Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
            for doc in docs_data
        ]
        
        logger.info(f"Index loaded from {path} with {len(self.documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve most relevant documents for query"""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")
        
        # Embed query
        query_embedding = self.embed_text(query).reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = 1 / (1 + dist)  # Convert distance to similarity score
                
                if score_threshold is None or score >= score_threshold:
                    results.append((doc, score))
        
        return results
    
    def format_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """Format retrieved documents as context for AI"""
        if not retrieved_docs:
            return ""
        
        context = "Konteksti (haettu tietokannasta):\n\n"
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            context += f"[{i}] (relevanssi: {score:.2f})\n"
            context += f"{doc.content}\n\n"
        
        return context
    
    def add_documents(self, new_documents: List[Document]):
        """Add new documents to existing index"""
        if self.index is None:
            self.build_index(new_documents)
        else:
            # Embed new documents
            new_documents = self.embed_documents(new_documents)
            
            # Add to index
            embeddings = np.array([doc.embedding for doc in new_documents])
            self.index.add(embeddings)
            
            # Add to documents list
            self.documents.extend(new_documents)
            
            logger.info(f"Added {len(new_documents)} documents. Total: {len(self.documents)}")


# Example usage
if __name__ == "__main__":
    # Create sample documents
    documents = [
        Document(
            id="1",
            content="Ajatuskumppani on suomalainen avoimen lähdekoodin tekoälyalusta.",
            metadata={"source": "about", "language": "fi"}
        ),
        Document(
            id="2",
            content="DePIN (Decentralized Physical Infrastructure Network) on hajautettu infrastruktuuriverkko.",
            metadata={"source": "tech", "language": "fi"}
        ),
        Document(
            id="3",
            content="Solana on nopea ja halpa blockchain-alusta.",
            metadata={"source": "blockchain", "language": "fi"}
        ),
    ]
    
    # Initialize retriever
    retriever = AjatusRAGRetriever()
    retriever.load_embedding_model()
    
    # Build index
    retriever.build_index(documents)
    
    # Test retrieval
    query = "Mikä on Ajatuskumppani?"
    results = retriever.retrieve(query, top_k=2)
    
    print(f"Query: {query}\n")
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Content: {doc.content}\n")
    
    # Format context
    context = retriever.format_context(results)
    print(context)
    
    # Save index
    retriever.save_index()

