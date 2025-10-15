"""
Main application script for Modern Multi-Modal RAG
"""
from pathlib import Path
from typing import List, Tuple
from .config import Config
from .document_processor import DocumentProcessor
from .summarizer import ContentSummarizer
from .retriever import ModernMultiVectorRetriever
from .qa_chain import ModernQAChain


class MultiModalRAGApp:
    """Main application class orchestrating the multimodal RAG pipeline"""
    
    def __init__(self, config_path: str = None):
        """Initialize the application with configuration"""
        print("🚀 Initializing Modern Multi-Modal RAG Application...")
        
        # Load configuration
        self.config = Config()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.summarizer = ContentSummarizer(self.config)
        self.retriever = ModernMultiVectorRetriever(self.config)
        self.qa_chain = None  # Will be initialized after processing documents
        
        print("✅ Application initialized successfully")
    
    def process_document(self, pdf_path: str = None) -> Tuple[List[str], List[str], List[str]]:
        """Process a PDF document and extract multimodal elements"""
        print("\n" + "="*60)
        print("📄 DOCUMENT PROCESSING PHASE")
        print("="*60)
        
        # Use provided path or default from config
        if pdf_path:
            pdf_file = Path(pdf_path)
        else:
            pdf_file = self.config.get_pdf_path()
        
        # Extract elements from PDF
        raw_elements = self.document_processor.extract_elements_from_pdf(pdf_file)
        
        # Categorize elements
        text_elements, table_elements, image_elements = self.document_processor.categorize_elements(raw_elements)
        
        return text_elements, table_elements, image_elements
    
    def create_summaries(self, text_elements: List[str], table_elements: List[str], 
                        image_elements: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Create summaries for all element types"""
        print("\n" + "="*60)
        print("🧠 SUMMARIZATION PHASE")
        print("="*60)
        
        # Create summaries
        text_summaries = self.summarizer.summarize_text_elements(text_elements)
        table_summaries = self.summarizer.summarize_table_elements(table_elements)
        image_summaries = self.summarizer.summarize_image_elements(image_elements)
        
        return text_summaries, table_summaries, image_summaries
    
    def setup_retrieval(self, text_summaries: List[str], text_elements: List[str],
                       table_summaries: List[str], table_elements: List[str],
                       image_summaries: List[str]) -> None:
        """Setup the retrieval system with processed content"""
        print("\n" + "="*60)
        print("🔍 RETRIEVAL SETUP PHASE")
        print("="*60)
        
        # Add all content to retriever
        self.retriever.add_all_content(
            text_summaries, text_elements,
            table_summaries, table_elements,
            image_summaries
        )
        
        # Initialize Q&A chain
        self.qa_chain = ModernQAChain(self.config, self.retriever)
        
        # Print retriever statistics
        stats = self.retriever.get_stats()
        print(f"📊 Retriever Statistics: {stats}")
    
    def run_demo_queries(self) -> None:
        """Run a set of demo queries to showcase the application"""
        print("\n" + "="*60)
        print("💬 DEMO Q&A PHASE")
        print("="*60)
        
        if not self.qa_chain:
            print("⚠️  Q&A chain not initialized. Run process_document and setup_retrieval first.")
            return
        
        # Demo questions
        demo_questions = [
            "What do you see in the images?",
            "What is the name of the company?",
            "What is the product displayed in the image?",
            "How much are the total expenses of the company?",
            "What is the ROI?",
            "How much did the company sell in 2023?",
            "And in 2022?"
        ]
        
        print(f"🎯 Running {len(demo_questions)} demo questions...")
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 50)
            
            result = self.qa_chain.ask(question)
            print(f"🤖 Answer: {result['answer']}")
            
            if result.get('error'):
                print("⚠️  Error occurred during processing")
    
    def ask_question(self, question: str) -> dict:
        """Ask a single question"""
        if not self.qa_chain:
            return {"error": "Q&A chain not initialized"}
        
        return self.qa_chain.ask(question)
    
    def run_full_pipeline(self, pdf_path: str = None) -> None:
        """Run the complete multimodal RAG pipeline"""
        print("🔄 Starting Full Multi-Modal RAG Pipeline")
        print("="*70)
        
        try:
            # Step 1: Process document
            text_elements, table_elements, image_elements = self.process_document(pdf_path)
            
            # Step 2: Create summaries
            text_summaries, table_summaries, image_summaries = self.create_summaries(
                text_elements, table_elements, image_elements
            )
            
            # Step 3: Setup retrieval
            self.setup_retrieval(
                text_summaries, text_elements,
                table_summaries, table_elements,
                image_summaries
            )
            
            # Step 4: Run demo queries
            self.run_demo_queries()
            
            print("\n" + "="*70)
            print("✅ Pipeline completed successfully!")
            print("💡 You can now ask questions using ask_question()")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for the application"""
    try:
        # Initialize and run the application
        app = MultiModalRAGApp()
        app.run_full_pipeline()
        
        # Interactive mode
        print("\n🎮 Interactive Mode - Ask your own questions!")
        print("(Type 'quit' to exit)")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if question:
                result = app.ask_question(question)
                print(f"\n🤖 Answer: {result.get('answer', 'No answer generated')}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()