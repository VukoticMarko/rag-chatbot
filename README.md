# RAG Chatbot with Mistral-7B & PDF Intelligence

## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot powered by Mistral-7B that answers questions based on your custom PDF documents. The system combines document intelligence with large language models to deliver accurate, context-aware responses with source citations.


## Key Features
- Document Intelligence: Processes and understands your custom PDF content
- Structured Responses: Answers with summary, explanation, examples, and sources
- Local Execution: Runs entirely on your machine (GPU acceleration supported)
- Source Verification: Cites exact page numbers from your documents
- Context-Aware: Uses Maximum Marginal Relevance for optimal context retrieval

## Technical Stack
Component              | Technology                  
-----------------------|-----------------------------
Large Language Model | Mistral-7B-Instruct (4-bit quantized)
Vector Database    | ChromaDB                    
Embeddings         | all-mpnet-base-v2           
Framework          | LangChain                   
PDF Processing     | PyPDFLoader                 

## Installation
1. Clone repository:
``` bash
git clone https://github.com/VukoticMarko/rag-chatbot.git
cd rag-chatbot
```

2. Create virtual environment (recommended):
``` python
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.\.venv\Scripts\activate  # Windows
```

4. Install dependencies:
``` bash
pip install -r requirements.txt
```

6. Download Mistral-7B model (4-bit quantized):
``` bash
mkdir -p models
wget -P models https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

## Project Structure
```
rag-chatbot/
├── data/                   # Place your PDF documents here
├── models/                 # Mistral-7B model location
├── vector_db/              # Vector database storage
├── build_vector_db.py      # Document ingestion script
├── chat.py                 # Main chatbot interface
├── config.py               # Configuration settings
├── llm.py                  # LLM loading and setup
├── load_documents.py       # PDF processing
├── rag.py                  # RAG pipeline
├── vector_db.py            # Vector database management
└── requirements.txt        # Dependencies
```

## Usage
1. Prepare Your Documents:
Place PDF files in the data/ directory

2. Start Chatbot (automatically preapares vectors and models): 
python chatbot.py

Example interaction:
Question: Explain attention mechanism in transformers =>
[Structured response with technical breakdown and sources]

## Configuration (config.py)
# Model paths
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Embedding model
```
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

# Document processing
```
CHUNK_SIZE = 1000    # Size of text chunks
CHUNK_OVERLAP = 200  # Overlap between chunks
```

# Vector database
```
VECTOR_DB_DIR = "vector_db"
```

## Response Format
The chatbot provides structured responses with (template can be changed):
```
1. Summary
   - Concise 2-sentence overview

2. Core Explanation
   - Detailed technical breakdown
   - Mathematical notation: E=mc²

3. Key Concepts
   - Concept 1: Definition
   - Concept 2: Definition

4. Examples/Applications
   # Practical code example
   def attention(q, k, v):
       return softmax(q @ k.T) @ v
   - Real-world use case 1
   - Real-world use case 2

5. Limitations
   - When not to apply this
   - Edge cases

6. Sources
   - Page 42: Attention Mechanisms Explained
   - Page 87: Transformer Architectures
```
## License
MIT License - Free for academic and commercial use with attribution
