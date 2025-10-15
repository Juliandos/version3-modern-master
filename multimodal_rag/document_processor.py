"""
Document processing module for extracting text, tables, and images from PDFs
"""
import base64
import os
from pathlib import Path
from typing import List, Tuple, Any
from unstructured.partition.pdf import partition_pdf
import pytesseract
from .config import Config


class DocumentProcessor:
    """Handles PDF document processing and element extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_tesseract()
        
    def _setup_tesseract(self) -> None:
        """Setup tesseract OCR engine"""
        tesseract_path = self.config.validate_tesseract()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✓ Tesseract found at: {tesseract_path}")
        else:
            print("⚠️  Warning: Tesseract not found. Image OCR may not work properly.")
            print("Please install tesseract:")
            print("  - macOS: brew install tesseract")
            print("  - Ubuntu: sudo apt install tesseract-ocr")
            print("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            
    def extract_elements_from_pdf(self, pdf_path: Path) -> List[Any]:
        """Extract all elements (text, tables, images) from PDF"""
        print(f"📄 Processing PDF: {pdf_path}")
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Ensure output directory exists
        self.config.setup_directories()
        
        try:
            raw_pdf_elements = partition_pdf(
                filename=str(pdf_path),
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=self.config.max_characters,
                new_after_n_chars=self.config.new_after_n_chars,
                combine_text_under_n_chars=self.config.combine_text_under_n_chars,
                image_output_dir_path=str(self.config.output_path),
            )
            print(f"✓ Successfully extracted {len(raw_pdf_elements)} elements from PDF")
            return raw_pdf_elements
            
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF: {str(e)}")
    
    def categorize_elements(self, raw_elements: List[Any]) -> Tuple[List[str], List[str], List[str]]:
        """Categorize elements into text, tables, and images"""
        text_elements = []
        table_elements = []
        image_elements = []
        
        # Process text and table elements
        for element in raw_elements:
            element_type = str(type(element))
            if 'CompositeElement' in element_type:
                text_elements.append(element.text)
            elif 'Table' in element_type:
                table_elements.append(element.text)
        
        # Process image elements from figures directory
        if self.config.output_path.exists():
            for image_file in self.config.output_path.iterdir():
                if image_file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    try:
                        encoded_image = self._encode_image(image_file)
                        image_elements.append(encoded_image)
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to encode image {image_file}: {e}")
        
        print(f"📊 Categorized elements:")
        print(f"  - Text elements: {len(text_elements)}")
        print(f"  - Table elements: {len(table_elements)}")
        print(f"  - Image elements: {len(image_elements)}")
        
        return text_elements, table_elements, image_elements
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')