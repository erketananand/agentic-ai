# Chunking Module
# Provides multiple strategies for splitting code into meaningful chunks

import ast
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_code_recursive(docs: list, chunk_size: int = 1200, chunk_overlap: int = 32) -> list:
    """
    Split code documents into smaller chunks using recursive character-based splitting.
    Uses Python-aware splitting to maintain code structure (functions, classes).

    This approach is faster but may split logical units across chunks.

    Args:
        docs: List of Document objects to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks to maintain context

    Returns:
        List of chunked Document objects
    """
    # Create a splitter that understands Python syntax
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,  # Max characters per chunk
        chunk_overlap=chunk_overlap,  # Overlap between chunks to maintain context
    )
    # Split all documents into chunks
    return splitter.split_documents(docs)


def chunk_code_ast(docs: list) -> list:
    """
    Parse Python code using Abstract Syntax Tree (AST) to create semantically meaningful chunks.
    Each chunk represents a complete class or module-level function.
    This preserves code structure better than fixed-size character splitting.

    This approach is slower but creates more semantically coherent chunks.

    Args:
        docs: List of Document objects to chunk

    Returns:
        List of chunked Document objects with type and name metadata
    """
    chunks = []

    for doc in docs:
        # Extract source file path and code content
        source = doc.metadata.get("source", "unknown")
        code = doc.page_content

        # Parse code into an AST tree; if syntax error, add the whole file as a chunk
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # File has syntax errors, treat as single chunk
            chunks.append(doc)
            continue

        # Add parent references — ast.walk doesn't expose parents by default
        # This lets us identify which nodes are top-level (direct children of Module)
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node

        # Iterate through all AST nodes to extract top-level classes and functions
        for node in ast.walk(tree):
            parent = getattr(node, "parent", None)

            # Extract top-level class definitions (classes directly in the module)
            # Include all methods and attributes within the class
            if isinstance(node, ast.ClassDef) and isinstance(parent, ast.Module):
                text = ast.get_source_segment(code, node)
                if text:
                    chunks.append(Document(
                        page_content=text,
                        metadata={"source": source, "type": "class", "name": node.name},
                    ))

            # Extract module-level function definitions (not methods inside a class)
            # Includes both regular and async functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(parent, ast.Module):
                text = ast.get_source_segment(code, node)
                if text:
                    chunks.append(Document(
                        page_content=text,
                        metadata={"source": source, "type": "function", "name": node.name},
                    ))

    return chunks
