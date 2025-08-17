from langchain_community.vectorstores import Chroma
from langchain.schema import Document as LangChainDocument # Import with an alias for clarity
from .load import load_documents
from .chunk import chunk_documents
from core.llm_provider import embedding_model

# Define a persistent path for the Chroma database
CHROMA_PATH = "chroma_db"

def run_ingestion_pipeline():
    """
    Runs the full ingestion pipeline:
    1. Loads documents from LlamaParse.
    2. Converts them to the LangChain format.
    3. Chunks the documents.
    4. Creates embeddings and stores them in ChromaDB.
    """
    # 1. Load documents using your load.py script (returns LlamaIndex Documents)
    llama_parse_docs = load_documents()
    if not llama_parse_docs:
        print("No documents were loaded. Exiting.")
        return
        
    # 2. CONVERT from LlamaIndex format to LangChain format
    # The text splitter expects documents with a 'page_content' attribute
    docs_to_chunk = [
        LangChainDocument(page_content=doc.text, metadata=doc.metadata) 
        for doc in llama_parse_docs
    ]

    # 3. Chunk the converted documents
    chunked_docs = chunk_documents(docs_to_chunk)
    
    # 4. Create or update the vector store
    print("Creating/updating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    
    # Ensure data is saved to disk
    vector_store.persist()
    
    print("\n✅ Ingestion complete!")
    print(f"Vector store created at: {CHROMA_PATH}")