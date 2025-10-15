"""
Modern MultiVector Retriever implementation using LangChain v0.3+ patterns
"""
import uuid
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from .config import Config


class ModernMultiVectorRetriever:
    """Modern implementation of MultiVector retriever with improved patterns"""
    
    def __init__(self, config: Config):
        self.config = config
        self.id_key = "doc_id"
        
        # Initialize components using modern patterns
        self._setup_retriever()
        
    def _setup_retriever(self) -> None:
        """Setup the retriever components"""
        try:
            # Initialize embedding model
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  # Latest embedding model
            )
            
            # Initialize vector store
            self.vectorstore = Chroma(
                collection_name="multimodal_summaries",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"  # Persist to disk
            )
            
            # Initialize document store
            self.docstore = InMemoryStore()
            
            # Create the MultiVectorRetriever
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                id_key=self.id_key,
                search_kwargs={"k": 4}  # Return top 4 results
            )
            
            print("✓ MultiVector retriever initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {str(e)}")
    
    def add_documents_to_retriever(self, summaries: List[str], original_contents: List[str], content_type: str) -> None:
        """Add documents to the retriever with improved error handling"""
        if not summaries or not original_contents:
            print(f"⚠️  No {content_type} content to add")
            return
            
        if len(summaries) != len(original_contents):
            raise ValueError(f"Mismatch in {content_type} summaries and original content lengths")
        
        try:
            # Generate unique IDs for each document
            doc_ids = [str(uuid.uuid4()) for _ in summaries]
            
            # Create summary documents with metadata
            summary_docs = [
                Document(
                    page_content=summary,
                    metadata={
                        self.id_key: doc_ids[i],
                        "content_type": content_type,
                        "index": i
                    }
                )
                for i, summary in enumerate(summaries)
            ]
            
            # Add summaries to vector store using modern method
            self.vectorstore.add_documents(summary_docs)
            
            # Store original contents in docstore
            doc_pairs = list(zip(doc_ids, original_contents))
            self.docstore.mset(doc_pairs)
            
            print(f"✓ Added {len(summaries)} {content_type} documents to retriever")
            
        except Exception as e:
            print(f"⚠️  Error adding {content_type} documents: {e}")
            raise
    
    def add_all_content(self, text_summaries: List[str], text_elements: List[str],
                       table_summaries: List[str], table_elements: List[str],
                       image_summaries: List[str]) -> None:
        """Add all content types to the retriever"""
        print("📚 Adding all content to retriever...")
        
        # Add text content
        if text_summaries and text_elements:
            self.add_documents_to_retriever(text_summaries, text_elements, "text")
        
        # Add table content
        if table_summaries and table_elements:
            self.add_documents_to_retriever(table_summaries, table_elements, "table")
        
        # Add image content (summaries are stored as original content too)
        if image_summaries:
            self.add_documents_to_retriever(image_summaries, image_summaries, "image")
        
        print("✅ All content added to retriever successfully")
    
    def search(self, query: str, k: int = 4) -> List[str]:
        """Search for relevant documents using modern invoke method"""
        try:
            # Using modern invoke method instead of deprecated get_relevant_documents
            results = self.retriever.invoke(query)
            
            # Extract content from results
            if isinstance(results, list):
                return [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in results[:k]]
            else:
                return [str(results)]
                
        except Exception as e:
            print(f"⚠️  Error during search: {e}")
            return [f"Search error: {str(e)}"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever"""
        try:
            # Get vector store statistics
            collection = self.vectorstore.get()
            vector_count = len(collection.get('ids', []))
            
            # Get docstore statistics  
            docstore_keys = list(self.docstore.yield_keys())
            docstore_count = len(docstore_keys)
            
            return {
                "vector_documents": vector_count,
                "stored_documents": docstore_count,
                "collection_name": "multimodal_summaries"
            }
        except Exception as e:
            return {"error": str(e)}