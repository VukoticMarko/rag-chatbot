import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

DATA_DIR = "data"

def load_and_chunk_documents():
    all_docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"Loading: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                all_docs.extend(docs)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    
    print(f"Loaded {len(all_docs)} pages")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_documents(all_docs)
