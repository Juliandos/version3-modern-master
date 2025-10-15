"""
Configuration management for the Multi-Modal RAG application
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

class Config:
    """Configuration class for managing environment variables and settings"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv(find_dotenv())
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # LangSmith configuration (optional)
        self.langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "multimodal-rag-modern")
        
        # Application settings
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
        self.max_characters = int(os.getenv("MAX_CHARACTERS", "4000"))
        self.new_after_n_chars = int(os.getenv("NEW_AFTER_N_CHARS", "3800"))
        self.combine_text_under_n_chars = int(os.getenv("COMBINE_TEXT_UNDER_N_CHARS", "2000"))
        
        # Model configurations
        self.gpt_35_model = os.getenv("GPT_35_MODEL", "gpt-3.5-turbo")
        self.gpt_4o_model = os.getenv("GPT_4O_MODEL", "gpt-4o")
        
        # Paths
        self.input_path = Path(os.getenv("INPUT_PATH", os.getcwd()))
        self.output_path = self.input_path / "figures"
        self.pdf_filename = os.getenv("PDF_FILENAME", "startupai-financial-report-v2.pdf")
        
    def setup_directories(self) -> None:
        """Create necessary directories"""
        self.output_path.mkdir(exist_ok=True)
        
    def get_pdf_path(self) -> Path:
        """Get the full path to the PDF file"""
        return self.input_path / self.pdf_filename
        
    def validate_tesseract(self) -> Optional[str]:
        """Validate tesseract installation and return path if found"""
        import shutil
        import platform
        
        # Try to find tesseract in system PATH first
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd:
            return tesseract_cmd
            
        # Platform-specific fallback paths
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
        elif platform.system() == "Darwin":  # macOS
            common_paths = [
                "/opt/homebrew/bin/tesseract",
                "/usr/local/bin/tesseract"
            ]
        else:  # Linux
            common_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract"
            ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
                
        return None