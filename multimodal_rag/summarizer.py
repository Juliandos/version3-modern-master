"""
Content summarization module using modern LangChain patterns
"""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import Config


class ContentSummarizer:
    """Handles summarization of text, tables, and images using LLM models"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize LLM chains using modern LangChain patterns
        self.text_chain = ChatOpenAI(
            model=config.gpt_35_model,
            max_tokens=config.max_tokens,
            temperature=0
        )
        
        self.vision_chain = ChatOpenAI(
            model=config.gpt_4o_model,
            max_tokens=config.max_tokens,
            temperature=0
        )
        
        # Create prompt templates using LCEL
        self.text_prompt = ChatPromptTemplate.from_template(
            "Summarize the following text concisely, focusing on key information:\n\n{text}\n\nSummary:"
        )
        
        self.table_prompt = ChatPromptTemplate.from_template(
            "Summarize the following table, highlighting key data points and relationships:\n\n{table}\n\nSummary:"
        )
        
        # Create chains using LCEL (LangChain Expression Language)
        self.text_summarizer = self.text_prompt | self.text_chain | StrOutputParser()
        self.table_summarizer = self.table_prompt | self.text_chain | StrOutputParser()
        
    def summarize_text_elements(self, text_elements: List[str]) -> List[str]:
        """Summarize text elements using GPT-3.5"""
        print(f"🔤 Summarizing {len(text_elements)} text elements...")
        
        summaries = []
        for i, text in enumerate(text_elements, 1):
            try:
                # Using modern invoke method instead of deprecated approaches
                summary = self.text_summarizer.invoke({"text": text})
                summaries.append(summary)
                print(f"  ✓ Text element {i} processed")
            except Exception as e:
                print(f"  ⚠️  Error processing text element {i}: {e}")
                summaries.append(f"Error processing text: {str(e)[:100]}...")
        
        return summaries
    
    def summarize_table_elements(self, table_elements: List[str]) -> List[str]:
        """Summarize table elements using GPT-3.5"""
        print(f"📋 Summarizing {len(table_elements)} table elements...")
        
        summaries = []
        for i, table in enumerate(table_elements, 1):
            try:
                # Using modern invoke method
                summary = self.table_summarizer.invoke({"table": table})
                summaries.append(summary)
                print(f"  ✓ Table element {i} processed")
            except Exception as e:
                print(f"  ⚠️  Error processing table element {i}: {e}")
                summaries.append(f"Error processing table: {str(e)[:100]}...")
                
        return summaries
    
    def summarize_image_elements(self, image_elements: List[str]) -> List[str]:
        """Summarize image elements using GPT-4o vision capabilities"""
        print(f"🖼️  Summarizing {len(image_elements)} image elements...")
        
        summaries = []
        for i, encoded_image in enumerate(image_elements, 1):
            try:
                # Modern multimodal approach with proper message structure
                messages = [
                    SystemMessage(content="You are an expert at analyzing images and describing their contents concisely."),
                    HumanMessage(content=[
                        {
                            "type": "text",
                            "text": "Describe the contents of this image in detail, focusing on key visual elements, text, charts, or data presented."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "high"
                            }
                        }
                    ])
                ]
                
                # Using modern invoke method instead of deprecated approaches
                response = self.vision_chain.invoke(messages)
                summaries.append(response.content)
                print(f"  ✓ Image element {i} processed")
                
            except Exception as e:
                print(f"  ⚠️  Error processing image element {i}: {e}")
                summaries.append(f"Error processing image: {str(e)[:100]}...")
                
        return summaries