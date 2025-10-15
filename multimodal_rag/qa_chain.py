"""
Modern Q&A Chain implementation using LangChain v0.3+ LCEL patterns
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from .config import Config
from .retriever import ModernMultiVectorRetriever


class ModernQAChain:
    """Modern Q&A chain using LCEL patterns for multimodal RAG"""
    
    def __init__(self, config: Config, retriever: ModernMultiVectorRetriever):
        self.config = config
        self.retriever = retriever
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=config.gpt_35_model,
            temperature=0,
            max_tokens=config.max_tokens
        )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt())
        ])
        
        # Build the chain using LCEL
        self._build_chain()
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Q&A chain"""
        return """You are an expert assistant that answers questions based on multimodal content including text, tables, and images.

Guidelines:
- Answer questions based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific and cite relevant details from the context
- For questions about images, refer to the image descriptions provided
- For questions about data, reference tables and numerical information when available
- Provide concise but comprehensive answers"""

    def _get_human_prompt(self) -> str:
        """Get the human prompt template"""
        return """Context (includes text, tables, and image descriptions):
{context}

Question: {question}

Answer:"""

    def _build_chain(self) -> None:
        """Build the Q&A chain using modern LCEL patterns"""
        # Create the retrieval function
        def format_docs(docs):
            """Format retrieved documents for context"""
            if isinstance(docs, list):
                return "\n\n".join([
                    f"Content: {doc}" for doc in docs
                ])
            return str(docs)
        
        # Build chain using LCEL
        self.chain = (
            RunnableParallel({
                "context": lambda x: format_docs(self.retriever.search(x["question"])),
                "question": RunnablePassthrough()
            })
            | RunnablePassthrough.assign(
                context=lambda x: x["context"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ Modern Q&A chain built successfully")
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with metadata"""
        try:
            print(f"🤔 Processing question: {question}")
            
            # Get the answer using modern invoke method
            answer = self.chain.invoke({"question": question})
            
            # Get retrieval context for transparency
            context_docs = self.retriever.search(question)
            
            result = {
                "question": question,
                "answer": answer,
                "context_used": len(context_docs),
                "retriever_stats": self.retriever.get_stats()
            }
            
            print(f"✅ Answer generated (used {len(context_docs)} context documents)")
            return result
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"⚠️  {error_msg}")
            return {
                "question": question,
                "answer": error_msg,
                "context_used": 0,
                "error": True
            }
    
    def batch_ask(self, questions: list) -> list:
        """Process multiple questions in batch"""
        print(f"📝 Processing {len(questions)} questions in batch...")
        
        try:
            # Use batch processing for efficiency
            inputs = [{"question": q} for q in questions]
            answers = self.chain.batch(inputs)
            
            results = []
            for i, (question, answer) in enumerate(zip(questions, answers)):
                results.append({
                    "question": question,
                    "answer": answer,
                    "batch_index": i
                })
                
            print(f"✅ Batch processing completed")
            return results
            
        except Exception as e:
            print(f"⚠️  Batch processing error: {e}")
            return [{"question": q, "answer": f"Batch error: {str(e)}", "error": True} for q in questions]