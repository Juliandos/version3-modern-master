# 🚀 Modern Multi-Modal RAG Application

A modernized version of the Multi-Modal RAG (Retrieval-Augmented Generation) application, upgraded to use the latest technologies including Python 3.13.3, Poetry 2.1.4, and LangChain v0.3+.

## 🎯 What This Application Does

This application processes PDF documents containing text, tables, and images, then allows you to ask questions about any of this multimodal content using natural language.

**Key Features:**
- 📄 Extract text, tables, and images from PDF files
- 🧠 Generate AI summaries of all content types
- 🔍 Vector-based semantic search across all content
- 💬 Natural language Q&A interface
- 🌍 Cross-platform compatibility (Windows, macOS, Linux)
- ⚡ Modern LangChain v0.3 patterns and performance optimizations

## 🆚 What's New in This Version

This is a complete modernization of the original project. Key improvements:

- **Python 3.13.3** with latest performance improvements
- **LangChain v0.3+** with LCEL patterns and modern methods
- **Modular architecture** instead of single-file approach
- **Cross-platform compatibility** (no more Windows-only paths!)
- **Comprehensive error handling** and user feedback
- **Type hints throughout** for better code clarity
- **Persistent vector storage** for better performance

👀 **See the full comparison** in the included `comparison-modern-vs-legacy.ipynb` notebook!

## 🔧 Requirements

### System Dependencies
- **Python 3.13.3+**
- **Poetry 2.1.4+** for dependency management
- **Tesseract OCR** for image text extraction:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Poppler** for PDF processing:
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - Windows: Download from [GitHub](https://github.com/oschwartz10612/poppler-windows)

### API Keys
- **OpenAI API Key** (required for GPT models and embeddings)
- **LangSmith API Key** (optional, for tracing and debugging)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Navigate to the modern version
cd version3-modern

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for LangSmith tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=multimodal-rag-modern
```

### 3. Add Your PDF
Place your PDF file in the project directory, or update the configuration to point to your PDF file.

### 4. Run the Application

**Option A: Full Pipeline (Recommended for first run)**
```bash
poetry run python -m multimodal_rag.main
```

**Option B: As a module**
```bash
python -m multimodal_rag.main
```

**Option C: Using Poetry script**
```bash
poetry run run-multimodal
```

### 5. Interactive Usage

After the initial processing, you can ask questions like:
- \"What do you see in the images?\"
- \"What is the company's revenue?\"
- \"Summarize the key findings from the tables\"
- \"What products are shown in the images?\"

## 📊 Project Structure

```
version3-modern/
├── multimodal_rag/           # Main application package
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── document_processor.py # PDF processing logic
│   ├── summarizer.py        # Content summarization
│   ├── retriever.py         # Vector retrieval system
│   ├── qa_chain.py          # Q&A chain implementation
│   └── main.py              # Application orchestration
├── pyproject.toml           # Poetry configuration
├── README.md                # This file
├── comparison-modern-vs-legacy.ipynb  # Educational comparison
└── .env.example             # Environment template
```

## 🎓 Educational Resources

### For Students and Learners

1. **📓 Comparison Notebook**: Open `comparison-modern-vs-legacy.ipynb` to see detailed explanations of what changed and why

2. **🔍 Code Comments**: Every module is fully documented with docstrings explaining the purpose and usage

3. **📚 Modern Patterns**: Learn current best practices for:
   - LangChain v0.3 LCEL patterns
   - Modular Python application design
   - Cross-platform compatibility
   - Error handling and user feedback
   - Type hints and documentation

### Understanding the Flow

1. **Document Processing**: Extract text, tables, and images from PDF
2. **Summarization**: Create concise summaries using GPT models
3. **Vector Storage**: Store summaries as embeddings for semantic search
4. **Retrieval**: Find relevant content based on user questions
5. **Q&A Generation**: Generate natural language answers using context

## ⚙️ Configuration Options

You can customize the application behavior via environment variables:

```bash
# Model Configuration
GPT_35_MODEL=gpt-3.5-turbo
GPT_4O_MODEL=gpt-4o
MAX_TOKENS=1024

# Document Processing
MAX_CHARACTERS=4000
NEW_AFTER_N_CHARS=3800
COMBINE_TEXT_UNDER_N_CHARS=2000

# File Paths
INPUT_PATH=/path/to/your/documents
PDF_FILENAME=your-document.pdf
```

## 🐛 Troubleshooting

### Common Issues

**1. Tesseract not found**
```
⚠️ Warning: Tesseract not found. Image OCR may not work properly.
```
- **Solution**: Install tesseract using your system package manager
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

**2. OpenAI API Key error**
```
ValueError: OPENAI_API_KEY environment variable is required
```
- **Solution**: Add your OpenAI API key to the `.env` file

**3. PDF not found**
```
FileNotFoundError: PDF file not found: /path/to/file.pdf
```
- **Solution**: Ensure the PDF file exists in the specified location
- **Check**: The default filename is `startupai-financial-report-v2.pdf`

**4. Memory issues with large PDFs**
- **Solution**: Reduce the `MAX_CHARACTERS` setting in your `.env` file
- **Alternative**: Process the document in smaller chunks

### Getting Help

1. Check the error messages - they're designed to be helpful!
2. Review the comparison notebook for understanding the code structure
3. Ensure all system dependencies are installed
4. Verify your API keys are correct and have sufficient credits

## 🔄 Migration from Legacy Version

If you're upgrading from the original version:

1. **Review the comparison notebook** to understand the changes
2. **Update your dependencies** using the new `pyproject.toml`
3. **Update your code** to use the new modular structure
4. **Test cross-platform compatibility** if needed

## 🚦 Performance Tips

- **First run**: Initial processing takes time due to PDF parsing and summarization
- **Subsequent runs**: Vector database persists, making retrieval faster
- **Large PDFs**: Consider processing in batches or reducing content size
- **API costs**: Monitor your OpenAI usage, especially with GPT-4o for images

## 📈 What's Next?

This modernized version provides a solid foundation for:
- Adding new document formats
- Implementing different embedding strategies
- Building web interfaces
- Adding authentication and user management
- Scaling to handle multiple documents

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Experiment with different models
- Add new features
- Improve error handling
- Optimize performance
- Share your learnings!

## 📜 License

This project is for educational use. Please respect OpenAI's usage policies and ensure you have appropriate API access.

---

**Happy learning! 🎉**