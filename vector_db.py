from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma
from load_documents import load_and_chunk_documents
from config import EMBEDDING_MODEL, VECTOR_DB_DIR
import torch

# Verify CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

def get_vector_db():
    # Load and chunk documents
    chunks = load_and_chunk_documents()
    
    # Initialize embedding model with GPU support if available
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {}
    )
    print(f"Embedding a sample chunk: {chunks[0].page_content}")
    print(f"Embedding vector: {embeddings.embed_documents([chunks[0].page_content])}")

    
    # Create vector store using the Document objects directly
    vector_db = Chroma.from_documents(
        documents=chunks,  # Pass Document objects, not just text
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    return vector_db