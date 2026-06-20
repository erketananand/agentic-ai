# RAG System Suite

Comprehensive Retrieval-Augmented Generation implementations for code analysis with four distinct approaches, plus a hybrid system combining all three retrieval strategies.

## Quick Navigation

| System | File | Focus | Best For |
|--------|------|-------|----------|
| **Semantic RAG** | `1_semantic_rag/2_ast_semantic_rag.py` | Vector embeddings | Conceptual understanding, design patterns |
| **Lexical RAG** | `2_lexical_rag/2_lexical_rag_using_bm25_ast.py` | Keyword matching | Exact code locations, API lookups |
| **Hybrid (2-way)** | `4_hybrid_rag/1_hybrid_rag_with_semantic_lexical.py` | Semantic + Lexical | Balanced keyword + concept matching |
| **Graph RAG (LLM)** | `3_graph_rag/1_graph_rag_using_llm_graph_builder.py` | LLM-extracted graph | Rich relationships, high accuracy |
| **Graph RAG (Chunking)** | `3_graph_rag/2_graph_rag_using_chucking_graph_builder.py` | Deterministic graph | Fast graph building, zero LLM indexing |
| **Hybrid (3-way)** | `4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py` | All three combined | Comprehensive analysis ★ RECOMMENDED |

## Quick Start

```bash
# Try the recommended triple hybrid system
python 4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py

# Ask any question about the codebase
You: How does the auth module work and what depends on it?
```

## System Comparison Matrix

### Retrieval Strategy

| System | Index Type | Search Method | Speed | Best For |
|--------|-----------|---|---|---|
| Semantic | Vector embeddings | Cosine similarity | Medium | Patterns, concepts |
| Lexical | BM25 index | Term frequency-IDF | Fast | Exact identifiers |
| Graph | Knowledge graph | Ego graph traversal | Medium | Dependencies, architecture |
| Hybrid 2-way | Embeddings + BM25 | Ensemble RRF ranking | Medium | Balanced approach |
| Hybrid 3-way | All three combined | All + merge | Medium | Comprehensive analysis |

### Query Performance

| Query Type | Semantic | Lexical | Graph | Hybrid 2-way | Hybrid 3-way |
|-----------|----------|---------|-------|---|---|
| "How does X work?" | ✓✓✓ | - | - | ✓✓✓ | ✓✓✓ |
| "Where is function Y?" | - | ✓✓✓ | - | ✓✓ | ✓✓ |
| "What does X depend on?" | - | - | ✓✓✓ | - | ✓✓ |
| "Show me X and explain it" | ✓✓ | ✓✓ | - | ✓✓✓ | ✓✓✓ |
| Complex multi-part | ✓ | ✓ | ✓ | ✓✓ | ✓✓✓ |

## Project Structure

```
agentic-ai/
├── helpers/                    # Reusable utility modules (shared across projects)
│   ├── __init__.py
│   ├── code_loader.py          # Load Python files from repository
│   ├── chunking.py             # Code splitting strategies
│   ├── vector_store.py         # Embedding-based storage
│   └── retrievers.py           # BM25 retriever implementation
├── rag/                        # RAG implementations
│   ├── 1_semantic_rag.py          # Embedding-based semantic search
│   ├── 2_ast_semantic_rag.py       # AST-based semantic search
│   ├── 3_lexical_rag_using_bm25.py # Keyword-based lexical search
│   └── README.md               # This file
├── sample_project/             # Sample codebase for testing
└── requirements.txt            # Python dependencies
```

## RAG Implementations

### 1. Semantic RAG (`1_semantic_rag.py`)
**Strategy:** Embedding-based semantic search

**How it works:**
- Loads Python files and chunks them using recursive character-based splitting
- Converts chunks to embeddings using HuggingFace (all-MiniLM-L6-v2)
- Stores embeddings in Chroma vector database
- Retrieves semantically similar chunks based on query embeddings

**Pros:**
- Understands semantic meaning (not just keywords)
- Works well for conceptual queries
- Flexible chunk boundaries

**Cons:**
- Slower (needs embedding computation)
- Requires GPU for faster inference
- May miss exact keyword matches

**Configuration:**
```python
CHUNK_SIZE = 1200  # Adjust based on code complexity
```

---

### 2. AST Semantic RAG (`2_ast_semantic_rag.py`)
**Strategy:** AST-based semantic chunking with embeddings

**How it works:**
- Loads Python files and parses them using Abstract Syntax Tree (AST)
- Extracts complete classes and module-level functions as chunks
- Converts chunks to embeddings
- Retrieves relevant code units based on semantics

**Pros:**
- Creates semantically coherent chunks (complete functions/classes)
- Preserves code structure
- Better for understanding related code

**Cons:**
- Slower (AST parsing + embedding)
- Only works with valid Python syntax
- May create very large chunks for big classes

**Use case:**
```bash
python 2_ast_semantic_rag.py --repo /path/to/codebase
```

---

### 3. Lexical RAG using BM25 (`3_lexical_rag_using_bm25.py`)
**Strategy:** Keyword-based lexical search (no embeddings)

**How it works:**
- Loads Python files and chunks them using recursive character-based splitting
- Uses BM25 algorithm (Okapi variant) for keyword matching
- No embeddings—pure term frequency ranking
- Retrieves chunks with highest keyword overlap

**Pros:**
- Fastest (no embeddings needed)
- Works on CPU
- Exact keyword matching
- Great for code with consistent terminology

**Cons:**
- Doesn't understand semantic meaning
- Misses related concepts
- May struggle with synonyms

**Configuration:**
```python
CHUNK_SIZE = 1200  # Adjust for retrieval granularity
```

---

## Helper Modules

### `code_loader.py`
```python
from helpers import load_python_codebase

# Load all .py files from a repository
docs = load_python_codebase("/path/to/repo")
```

### `chunking.py`
```python
from helpers import chunk_code_recursive, chunk_code_ast

# Character-based chunking
chunks = chunk_code_recursive(docs, chunk_size=1200)

# AST-based chunking
chunks = chunk_code_ast(docs)
```

### `vector_store.py`
```python
from helpers import build_vector_store, load_vector_store

# In-memory vector store
vector_store = build_vector_store(chunks)

# Persistent vector store
vector_store = build_vector_store(chunks, persist_directory="./chroma_db")

# Load existing persistent store
vector_store = load_vector_store("./chroma_db")
```

### `retrievers.py`
```python
from helpers import BM25Retriever
from rank_bm25 import BM25Okapi

# Create BM25 retriever
tokenized = [doc.page_content.lower().split() for doc in chunks]
bm25_obj = BM25Okapi(tokenized)
retriever = BM25Retriever(docs=chunks, bm25=bm25_obj)
```

---

## Usage

### Run Semantic RAG
```bash
python 1_semantic_rag.py --repo ./sample_project
```

### Run AST Semantic RAG
```bash
python 2_ast_semantic_rag.py --repo ./sample_project
```

### Run Lexical RAG (BM25)
```bash
python 3_lexical_rag_using_bm25.py --repo ./sample_project
```

### Interactive Session
All three systems start an interactive Q&A session:
```
Ready. Ask your question. Type 'exit' to quit

You: How does authentication work?
Agent: [searches codebase and responds]

You: exit
```

---

## Comparison

| Feature | Semantic | AST Semantic | BM25 |
|---------|----------|--------------|------|
| Speed | Medium | Slow | **Fast** |
| Accuracy | High | **Highest** | Medium |
| GPU Required | Optional | Optional | No |
| Semantic Understanding | Yes | Yes | No |
| Chunk Coherence | Medium | **High** | Medium |
| Works Offline | Yes | Yes | Yes |
| Memory Usage | High | Medium | Low |

---

## Customization

### Adjust Chunk Size
Edit `CHUNK_SIZE` in each RAG file:
- **Smaller chunks** (256-512): More granular, faster retrieval, more redundancy
- **Larger chunks** (1000-2000): More context, slower retrieval, better coherence

### Use Different Embeddings
In `helpers/vector_store.py`, change the embedding model:
```python
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # More powerful
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"    # Lighter
```

### Use Different LLM
In RAG files, change the model:
```python
# From Groq
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Or use OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

---

## Requirements
```
langchain
langchain-chroma
langchain-groq
langchain-huggingface
langchain-core
langchain-text-splitters
rank-bm25
python-dotenv
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Environment Variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_TOKEN=your_huggingface_token  # Optional
```

---

## Tips

1. **Start with BM25** if speed is critical or running on CPU
2. **Use AST Semantic** for best code understanding
3. **Use Semantic** for more flexible, forgiving searches
4. **Adjust k in retriever_tool** to change number of results (default: 4)
5. **Experiment with chunk sizes** based on your codebase
6. **Use persistent vector stores** for large codebases to avoid re-computing
