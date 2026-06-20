# Vector Store Module
# Handles building and persisting vector databases for semantic search

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os


def build_vector_store(chunks: list, persist_directory: str = None, embedding_model: str = "all-MiniLM-L6-v2") -> Chroma:
    """
    Create a vector database (Chroma) from code chunks using embeddings.
    Embeddings convert code into numerical vectors for semantic similarity search.

    Args:
        chunks: List of Document objects to embed and store
        persist_directory: Optional path to persist the vector store to disk
                          If None, store is kept in-memory only
        embedding_model: HuggingFace embedding model name to use

    Returns:
        Chroma vector store instance (in-memory or persistent)

    Example:
        # In-memory vector store (lost when program exits)
        vector_store = build_vector_store(chunks)

        # Persistent vector store (saved to disk)
        vector_store = build_vector_store(chunks, persist_directory="./chroma_db")
    """
    if not chunks:
        raise ValueError("Cannot build vector store with empty chunks list")

    # Initialize HuggingFace embeddings using a lightweight, efficient model
    # Disable show_progress to reduce console noise
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        show_progress=False,
        encode_kwargs={"normalize_embeddings": True}
    )

    # Verify embeddings work by testing on first chunk
    try:
        test_embedding = embeddings.embed_query("test")
        if not test_embedding or len(test_embedding) == 0:
            raise ValueError("Embeddings model returned empty embedding")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

    # Create Chroma vector store with optional persistence
    try:
        if persist_directory:
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            # Persistent mode: saves to disk for reuse across sessions
            return Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # In-memory mode: faster but lost when program exits
            return Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )
    except Exception as e:
        raise RuntimeError(f"Failed to build vector store: {str(e)}")


def load_vector_store(persist_directory: str, embedding_model: str = "all-MiniLM-L6-v2") -> Chroma:
    """
    Load a previously persisted vector store from disk.

    Args:
        persist_directory: Path to the persisted vector store
        embedding_model: HuggingFace embedding model name (must match the original)

    Returns:
        Chroma vector store instance

    Example:
        # Load an existing persistent vector store
        vector_store = load_vector_store("./chroma_db")
    """
    if not os.path.exists(persist_directory):
        raise ValueError(f"Vector store directory does not exist: {persist_directory}")

    # Initialize the same embeddings model used during creation
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        show_progress=False,
        encode_kwargs={"normalize_embeddings": True}
    )
    # Load the persisted vector store
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
