# Document Processing and RAG System

This repository contains a comprehensive document processing and Retrieval-Augmented Generation (RAG) system. The system extracts text from various document formats (PDFs, PowerPoint presentations, videos) and creates a semantic search database that can be queried using state-of-the-art language models.

## System Overview

The project consists of two main components:

1. **Document Processor**: Extracts text content from different file types
2. **RAG System**: Creates embeddings, builds a vector database, and generates answers to queries using the extracted content

## Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- GPU recommended for faster processing (especially for the RAG component)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-processing-rag.git
cd document-processing-rag

# Install required packages
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# On Ubuntu/Debian:
apt install ffmpeg
# On macOS:
brew install ffmpeg
# On Windows:
# Download from https://ffmpeg.org/download.html
```

## File Structure

```
.
├── extract.py           # Document extraction script
├── rag.py               # RAG system implementation
├── requirements.txt     # Required Python packages
└── README.md            # This file
```

## Component 1: Document Processor (`extract.py`)

The document processor extracts text from:

- PDF files (using PyPDF2)
- PowerPoint presentations (using python-pptx)
- Video files (converting to audio and then transcribing using Google Speech Recognition)

### Features

- PDF text extraction
- PowerPoint text extraction (slides, notes, titles)
- Video-to-audio conversion
- Audio transcription
- Batch processing of multiple files

### Usage

```bash
# If running in Google Colab:
# 1. Upload the extract.py file
# 2. Configure the input and output paths in the script
# 3. Run the script

# If running locally:
python extract.py
```

### Configuration

Edit the following variables in `extract.py`:

```python
input_folder = "/path/to/your/input/folder"  # Directory containing your source files
output_folder = "/path/to/your/output/folder"  # Directory to save extracted text
```

## Component 2: RAG System (`rag.py`)

The RAG system:

1. Loads the extracted text files
2. Splits them into chunks for processing
3. Creates embeddings using Sentence Transformers
4. Builds a FAISS vector database for efficient similarity search
5. Provides an interface to query the database using Mistral-7B-Instruct

### Features

- Document chunking with overlap
- Embedding generation using Sentence Transformers
- Vector database creation with FAISS
- Interactive and batch query modes
- Language model integration (Mistral-7B-Instruct)
- Conversation history saving

### Usage

```bash
# First, make sure to run extract.py to create text files
# Then:
python rag.py
```

### Configuration

Edit the following variables in `rag.py`:

```python
INPUT_DIR = "/path/to/your/extracted/texts"  # Directory with extracted text files
VECTOR_DB_PATH = "/path/to/save/vector/database"  # Path to save the vector database
HF_TOKEN = "your_huggingface_token"  # Your Hugging Face token
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Model to use
```

## Using the RAG System

### Interactive Mode

Run the script and enter queries when prompted:

```
=== RAG System with Mistral-7B-Instruct-v0.3 Ready ===
Enter your query (or 'exit' to quit, 'save' to save conversation):

Query: What are the main concepts covered in these documents?
```

### Batch Mode

Edit the `queries` list in the script and set `mode = "batch"` to process multiple predefined queries:

```python
queries = [
    "What are the main concepts covered in these documents?",
    "Summarize the key points about topic X",
    "Explain concept Y in simple terms"
]
```

## Performance Tips

1. **Use a GPU** - Processing, especially for the RAG component, is significantly faster with GPU acceleration.
2. **Adjust chunk size** - Modify `CHUNK_SIZE` and `CHUNK_OVERLAP` based on your content structure.
3. **Choose the right model** - Smaller models (like Mistral-7B) offer good performance with lower memory requirements.
4. **Process in batches** - For very large document collections, process files in batches.

## Troubleshooting

### Common Issues

1. **Video transcription errors**:
   - Ensure good audio quality in videos
   - Try adjusting the chunk duration (`chunk_duration` in the `transcribe_audio` function)

2. **Out of memory errors**:
   - Reduce model size or use quantization options
   - Process smaller batches of documents

3. **FAISS errors**:
   - Ensure both faiss-cpu (or faiss-gpu) is correctly installed
   - Check for corrupted database files (delete and recreate if necessary)

4. **Hugging Face authentication issues**:
   - Verify your token has the necessary permissions
   - Ensure you've accepted the model terms on Hugging Face

## Customization

### Using Different Models

To use a different language model, change the `MODEL_ID` in `rag.py`:

```python
# Examples
MODEL_ID = "google/gemma-7b-it"  # Gemma 7B Instruct
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Mistral 7B
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2
```

### Changing Embedding Models

To use a different embedding model, modify the `HuggingFaceEmbeddings` initialization:

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Change model here
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
```

## License

[Insert your license information here]

## Acknowledgements

- PyPDF2 for PDF processing
- SpeechRecognition for audio transcription
- Hugging Face for model hosting
- LangChain for the RAG framework
- FAISS for vector similarity search
