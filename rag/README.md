# RAG System Suite

Comprehensive Retrieval-Augmented Generation implementations for code analysis with multiple approaches organized by retrieval strategy, plus hybrid systems combining different strategies.

## Quick Navigation

| System | File | Focus | Best For |
|--------|------|-------|----------|
| **Semantic RAG (Recursive)** | `1_semantic_rag/1_semantic_rag_using_recursive_text_splitter.py` | Vector embeddings + recursive chunking | Flexible, adaptive concept understanding |
| **Semantic RAG (AST)** | `1_semantic_rag/2_semantic_rag_using_ast.py` | Vector embeddings + AST chunking | Coherent code units, design patterns |
| **Lexical RAG (Recursive)** | `2_lexical_rag/1_lexical_rag_using_bm25_recursive_text_splitter.py` | BM25 keyword matching + recursive chunking | Fast, exact code locations |
| **Lexical RAG (AST)** | `2_lexical_rag/2_lexical_rag_using_bm25_ast.py` | BM25 keyword matching + AST chunking | Fast structured searches |
| **Graph RAG (Chunking)** | `3_graph_rag/1_graph_rag_using_chucking_graph_builder.py` | Deterministic graph from code structure | Fast, zero LLM indexing |
| **Graph RAG (LLM)** | `3_graph_rag/2_graph_rag_using_llm_graph_builder.py` | LLM-extracted knowledge graph | Rich relationships, high accuracy |
| **Hybrid (2-way)** | `4_hybrid_rag/1_hybrid_rag_with_semantic_lexical.py` | Semantic + Lexical combined | Balanced keyword + concept matching |
| **Hybrid (3-way)** | `4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py` | All three strategies combined | Comprehensive analysis ★ RECOMMENDED |

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
├── rag/                        # RAG implementations organized by strategy
│   ├── 1_semantic_rag/
│   │   ├── 1_semantic_rag_using_recursive_text_splitter.py
│   │   ├── 2_semantic_rag_using_ast.py
│   │   └── README.md
│   ├── 2_lexical_rag/
│   │   ├── 1_lexical_rag_using_bm25_recursive_text_splitter.py
│   │   ├── 2_lexical_rag_using_bm25_ast.py
│   │   └── README.md
│   ├── 3_graph_rag/
│   │   ├── 1_graph_rag_using_chucking_graph_builder.py
│   │   ├── 2_graph_rag_using_llm_graph_builder.py
│   │   ├── chunking_graph_cache.graphml
│   │   ├── llm_graph_cache.graphml
│   │   └── README.md
│   ├── 4_hybrid_rag/
│   │   ├── 1_hybrid_rag_with_semantic_lexical.py
│   │   ├── 2_hybrid_rag_with_semantic_lexical_graph.py
│   │   ├── hybrid_graph_cache.graphml
│   │   └── README.md
│   ├── QUESTIONS.md            # Example queries and use cases
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment variables (not in repo)
│   └── README.md               # This file
├── sample_project/             # Sample codebase for testing
└── requirements.txt            # Root-level Python dependencies
```

## RAG Implementations

### 1. Semantic RAG

#### 1.1. Using Recursive Text Splitter (`1_semantic_rag_using_recursive_text_splitter.py`)
**Strategy:** Embedding-based semantic search with recursive chunking

**How it works:**
- Loads Python files and chunks using recursive character-based splitting
- Converts chunks to embeddings using HuggingFace (all-MiniLM-L6-v2)
- Stores embeddings in Chroma vector database
- Retrieves semantically similar chunks based on query embeddings

**Pros:**
- Understands semantic meaning (not just keywords)
- Flexible chunk boundaries adapt to content
- Works well for conceptual queries

**Cons:**
- Slower (embedding computation required)
- GPU optional but recommended
- May miss exact keyword matches

---

#### 1.2. Using AST (`2_semantic_rag_using_ast.py`)
**Strategy:** AST-based semantic chunking with embeddings

**How it works:**
- Loads Python files and parses using Abstract Syntax Tree (AST)
- Extracts complete classes and module-level functions as chunks
- Converts chunks to embeddings
- Retrieves relevant code units based on semantics

**Pros:**
- Creates semantically coherent chunks (complete functions/classes)
- Preserves code structure integrity
- Better for understanding related code units

**Cons:**
- Slower (AST parsing + embedding computation)
- Only works with valid Python syntax
- May create very large chunks for complex classes

**Use case:**
```bash
python 1_semantic_rag/2_semantic_rag_using_ast.py --repo /path/to/codebase
```

---

### 2. Lexical RAG (BM25)

#### 2.1. Using Recursive Text Splitter (`1_lexical_rag_using_bm25_recursive_text_splitter.py`)
**Strategy:** Keyword-based lexical search with recursive chunking

**How it works:**
- Loads Python files and chunks using recursive character-based splitting
- Uses BM25 algorithm (Okapi variant) for keyword matching
- No embeddings—pure term frequency ranking
- Retrieves chunks with highest keyword overlap

**Pros:**
- Fastest retrieval (no embeddings needed)
- Works on CPU only
- Exact keyword matching
- Great for code with consistent terminology

**Cons:**
- Doesn't understand semantic meaning
- Misses related concepts
- May struggle with synonyms

---

#### 2.2. Using AST (`2_lexical_rag_using_bm25_ast.py`)
**Strategy:** Keyword-based lexical search with AST chunking

**How it works:**
- Parses Python files using Abstract Syntax Tree (AST)
- Extracts complete functions and classes as chunks
- Applies BM25 ranking to structured code chunks
- Retrieves relevant code units by keyword matching

**Pros:**
- Fast keyword retrieval on structured chunks
- No embeddings needed
- Better chunk coherence than recursive splitting

**Cons:**
- Limited to valid Python syntax
- Semantic meaning not understood
- May struggle with cross-cutting concerns

---

### 3. Graph RAG

#### 3.1. Using Chunking-based Graph Builder (`1_graph_rag_using_chucking_graph_builder.py`)
**Strategy:** Deterministic graph from code structure

**How it works:**
- Builds knowledge graph directly from code structure (AST)
- No LLM calls—deterministic extraction
- Extracts functions, classes, and their relationships
- Traverses graph to find related code units

**Pros:**
- Fastest graph building
- Zero LLM indexing costs
- Fully deterministic and reproducible
- Works offline

**Cons:**
- May miss implicit relationships
- Limited to syntactic relationships
- Graph quality depends on code structure

---

#### 3.2. Using LLM Graph Builder (`2_graph_rag_using_llm_graph_builder.py`)
**Strategy:** LLM-extracted knowledge graph

**How it works:**
- Uses LLM to extract semantic relationships from code
- Builds rich knowledge graph with implicit relationships
- LLM understands context beyond syntax
- Traverses graph to find semantically related code

**Pros:**
- Rich relationship extraction
- Understands implicit dependencies
- High accuracy for complex relationships

**Cons:**
- Slower (LLM calls for indexing)
- Requires LLM API access
- Higher costs

---

### 4. Hybrid RAG

#### 4.1. Semantic + Lexical (`1_hybrid_rag_with_semantic_lexical.py`)
**Strategy:** Combines vector embeddings and BM25 keyword search

**How it works:**
- Uses both semantic embeddings and BM25 ranking
- Combines results using ensemble ranking (Reciprocal Rank Fusion)
- Retrieves chunks that match both semantically and lexically

**Pros:**
- Balanced approach combining both strategies
- Better coverage (semantic + keyword)
- Handles both conceptual and exact matches

---

#### 4.2. Semantic + Lexical + Graph (`2_hybrid_rag_with_semantic_lexical_graph.py`)
**Strategy:** All three strategies combined ★ RECOMMENDED

**How it works:**
- Combines semantic embeddings, BM25 ranking, and graph traversal
- Retrieves from all three indexes
- Merges and re-ranks results using ensemble methods

**Pros:**
- Most comprehensive analysis
- Covers conceptual, lexical, and relational queries
- Best coverage for complex questions

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

### Run Recommended Hybrid System (3-way)
```bash
python 4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py --repo ./sample_project
```

### Run Semantic RAG
```bash
# With recursive text splitting
python 1_semantic_rag/1_semantic_rag_using_recursive_text_splitter.py --repo ./sample_project

# With AST-based chunking
python 1_semantic_rag/2_semantic_rag_using_ast.py --repo ./sample_project
```

### Run Lexical RAG (BM25)
```bash
# With recursive text splitting
python 2_lexical_rag/1_lexical_rag_using_bm25_recursive_text_splitter.py --repo ./sample_project

# With AST-based chunking
python 2_lexical_rag/2_lexical_rag_using_bm25_ast.py --repo ./sample_project
```

### Run Graph RAG
```bash
# With chunking-based graph builder (fast, no LLM)
python 3_graph_rag/1_graph_rag_using_chucking_graph_builder.py --repo ./sample_project

# With LLM-based graph builder (richer, slower)
python 3_graph_rag/2_graph_rag_using_llm_graph_builder.py --repo ./sample_project
```

### Run Hybrid RAG
```bash
# Semantic + Lexical
python 4_hybrid_rag/1_hybrid_rag_with_semantic_lexical.py --repo ./sample_project

# Semantic + Lexical + Graph (recommended)
python 4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py --repo ./sample_project
```

### Interactive Session
All systems start an interactive Q&A session:
```
Ready. Ask your question. Type 'exit' to quit

You: How does authentication work and what depends on it?
Agent: [searches codebase with appropriate strategy and responds]

You: Where is the user validation function?
Agent: [lexical search finds exact location]

You: exit
```

---

## Strategy Comparison

| Feature | Semantic Recursive | Semantic AST | BM25 Recursive | BM25 AST | Graph Chunking | Graph LLM | Hybrid 2-way | Hybrid 3-way |
|---------|----------|----------|---------|---------|---------|---------|---------|---------|
| Speed | Medium | Slow | **Fast** | **Fast** | Medium | Slow | Medium | Medium |
| Accuracy | High | **Highest** | Medium | Medium | High | **Highest** | Very High | **★ Best** |
| GPU Required | Optional | Optional | No | No | No | No | Optional | Optional |
| Semantic Understanding | Yes | Yes | No | No | Partial | Yes | Yes | **Yes** |
| Chunk Coherence | Medium | **High** | Medium | **High** | N/A | N/A | High | **High** |
| Works Offline | Yes | Yes | Yes | Yes | Yes | No | Yes | Partial |
| Memory Usage | High | Medium | Low | Low | Medium | High | High | Very High |
| Best For | Concepts | Code structure | Keywords | Keywords | Dependencies | Complex relations | Balanced | Everything ★ |

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
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
llm = ChatGroq(model="llama3-8b-8192", temperature=0)
llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# Or use OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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

## Quick Start Guide

**For the best overall experience:** Start with the recommended 3-way hybrid system:
```bash
python 4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py --repo ./sample_project
```

**Choose based on your priority:**

| Priority | System | Why |
|----------|--------|-----|
| **Speed** | BM25 (either version) | No embeddings needed, instant results |
| **Accuracy** | Hybrid 3-way | Best coverage of all retrieval strategies |
| **Code Understanding** | Semantic AST | Complete functions/classes preserved |
| **Exact Matches** | BM25 AST | Fast, structured keyword search |
| **Dependency Mapping** | Graph (chunking) | Fast relationship extraction |
| **Complex Relations** | Graph (LLM) | Semantic relationship understanding |

## Tips

1. **Start with 3-way Hybrid** for best overall results
2. **Use BM25** if speed is critical or running on CPU-only systems
3. **Use Semantic AST** for deep code understanding with structured chunks
4. **Use Graph RAG** when dependency/relationship questions are important
5. **Adjust k in retriever** to change number of results (default: 4)
6. **Experiment with chunk sizes** in code based on your codebase complexity
7. **Use persistent vector stores** for large codebases to avoid re-indexing
8. **Enable caching** in Graph RAG for faster subsequent queries (graphml files)
