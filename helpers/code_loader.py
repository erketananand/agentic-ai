# Code Loader Module
# Loads Python files from a repository into LangChain Document objects

from pathlib import Path
from langchain_core.documents import Document


def load_python_codebase(repo_path: str) -> list:
    """
    Load all Python files from a repository into Document objects.
    Each document contains the file's code and metadata about its source path.

    Args:
        repo_path: Path to the repository directory

    Returns:
        List of Document objects with code content and source metadata
    """
    docs = []
    # Recursively find all .py files in the repository
    for path in Path(repo_path).rglob("*.py"):
        # Read file content with UTF-8 encoding, ignoring decode errors
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Create a Document with code content and source file path as metadata
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs
