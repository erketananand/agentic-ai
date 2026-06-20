# Lexical RAG System

Two implementations of keyword-based retrieval systems using BM25 (Best Matching 25) algorithm for fast, efficient code search without embeddings. Both use LangChain agents to answer questions about Python codebases.

## Overview

**Lexical RAG** uses BM25 for term-based searching instead of vector embeddings. BM25 is a probabilistic ranking function that scores document relevance based on:
- **Term Frequency (TF)**: How often a term appears in the chunk
- **Inverse Document Frequency (IDF)**: How rare the term is across all chunks
- **Document Length Normalization**: Prevents longer documents from being ranked higher

This approach is **fast** (no embedding computation), **lightweight** (no model loading), and **interpretable** (see exactly why a result matched).

**Key Difference Between Files:**
- **File 1**: Fixed-size recursive character splitting (1200 chars per chunk)
- **File 2**: AST-based splitting (complete functions/classes per chunk) for better semantic structure

---

## 1_lexical_rag_using_bm25_recursive_text_splitter.py

**Simple character-based BM25 lexical search** using fixed-size chunks from recursive text splitting.

### How It Works

1. **Loading**: Reads all Python files from target directory
2. **Chunking**: Splits code recursively on characters into 1200-character chunks
   - Simple, predictable splits
   - May split functions/classes in the middle
   - Good for breadth of coverage
3. **Indexing**: Creates BM25 index by tokenizing all chunks
   - Converts each chunk to lowercase words (tokens)
   - Computes TF-IDF scores
   - Builds ranked inverted index
4. **Retrieval**: For each query, scores all chunks using BM25 algorithm
   - Returns top-k matching chunks
   - Fast lookup (no embeddings, no neural network inference)
5. **Answering**: LLM agent receives retrieved chunks, synthesizes answer

### BM25 Algorithm Explained

**The BM25 Formula** (simplified):
```
score(doc, query) = sum over query terms:
    IDF(term) * (TF(term, doc) * (k1 + 1)) / (TF(term, doc) + k1 * (1 - b + b * |doc| / avgdl))
```

Where:
- `IDF(term)` = log(N / df(term)) → rare terms score higher
- `TF(term, doc)` = term frequency in this document
- `k1` = saturation constant (default ~1.2, controls how much TF matters)
- `b` = length normalization (default ~0.75, prevents longer docs from winning)
- `|doc|` = document length
- `avgdl` = average document length

**Example**:
```
Query: "authenticate user login"
Document 1: "User login failed" (small, exact terms)
Document 2: "The authentication system verifies users during login process" (large, less dense)

BM25 scores:
- Document 1: Higher (dense matches, normalized)
- Document 2: Lower (matches but longer, normalized down)
```

### Usage Examples

```bash
# Default: 1200-char chunks
python 1_lexical_rag_using_bm25_recursive_text_splitter.py --repo ./my_project

# Custom repository
python 1_lexical_rag_using_bm25_recursive_text_splitter.py --repo ~/projects/backend

# Then ask questions interactively:
# You: Where is user authentication handled?
# You: Show me the payment processing logic
# You: What files handle database connections?
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `load_python_codebase()` | Recursively loads all .py files as LangChain Documents |
| `chunk_code_recursive()` | Splits documents recursively on character boundaries (fixed 1200-char chunks) |
| `build_bm25_retriever()` | Tokenizes chunks, creates BM25 index, returns retriever |
| `build_agent()` | Creates LangChain agent with retriever tool and safety middleware |
| `BM25Retriever` | LangChain-compatible wrapper around rank_bm25.BM25Okapi |

### Chunking Behavior

```
Original file (2500 chars):
┌─────────────────────────────────────────────────────────────────┐
│  def login(user):                                               │
│    verify(user)                                                 │
│    create_session(user)                                         │
│    return token                                                 │
│                                                                 │
│  def authenticate(token):                                       │
│    validate(token)                                              │
│    extract_user(token)                                          │
│                                                                 │
│  class UserManager:                                             │
│    def __init__(self):                                          │
│      self.sessions = {}                                         │
│      ...more code...                                            │
└─────────────────────────────────────────────────────────────────┘

CHUNK_SIZE = 1200:
┌────────────────────────────────────────────┐  ┌────────────────┐
│ Chunk 1 (chars 0-1200)                     │  │ Chunk 2        │
│ def login(user):                           │  │ (chars 1200+)  │
│   verify(user)                             │  │ def            │
│   create_session(user)                     │  │ authenticate.. │
│   return token                             │  │                │
│                                            │  │ ...more...     │
│ def authenticate(token):                   │  │                │
│   validate(token)                          │  │ (may split     │
│   extract_user(token)                      │  │ functions)     │
│ class UserManager:                         │  │                │
│   def __init__(self):                      │  └────────────────┘
│     self.sessions = {}                     │
│     ...                                    │
└────────────────────────────────────────────┘

Pros: Simple, predictable, covers all code
Cons: May cut functions mid-way, less semantic structure
```

### Strengths and Weaknesses

**Strengths:**
- ✅ **Fast**: No embeddings, instant indexing
- ✅ **Lightweight**: Minimal memory overhead
- ✅ **Interpretable**: Exact keyword matching is transparent
- ✅ **Deterministic**: Same query always returns same results
- ✅ **Good for identifiers**: Function names, variable names, imports

**Weaknesses:**
- ❌ **Semantic blindness**: Can't understand concept similarity ("login" ≠ "authenticate")
- ❌ **Chunking artifacts**: May split functions/classes mid-way
- ❌ **Synonym issues**: Misses "authenticate" when searching for "login"
- ❌ **Context loss**: Chunks might lack surrounding context

---

## 2_lexical_rag_using_bm25_ast.py

**Semantic-structure-aware BM25 lexical search** using AST (Abstract Syntax Tree) based chunking for complete functions and classes.

### How It Works

1. **Loading**: Reads all Python files
2. **AST Parsing**: Uses Python's AST module to extract code structure
   - Identifies classes, functions, methods
   - Each chunk is a **complete, syntactically valid** function or class
   - Preserves semantic boundaries
3. **Indexing**: Creates BM25 index (same as File 1)
4. **Retrieval**: BM25 scores chunks, returns matches
5. **Answering**: LLM synthesizes response from intact, coherent chunks

### AST Parsing Explained

**Abstract Syntax Tree** represents code structure as a tree:

```python
def login(user):
    verify(user)
    create_session(user)

class UserManager:
    def __init__(self):
        self.sessions = {}
```

Becomes:

```
Module
├── FunctionDef (name='login')
│   ├── Name (id='user')
│   ├── Call (func='verify')
│   └── Call (func='create_session')
└── ClassDef (name='UserManager')
    └── FunctionDef (name='__init__')
        └── Assign (target='self.sessions')
```

**Benefits for Chunking:**
- Each node = one complete, valid function/class
- No arbitrary character splits
- Chunk boundaries respect semantic structure
- Improved context preservation

### AST Chunking Behavior

```
Original file (2500 chars):
┌─────────────────────────────────────────────────────────────────┐
│  def login(user):                                               │
│    verify(user)                                                 │
│    create_session(user)                                         │
│    return token                                                 │
│                                                                 │
│  def authenticate(token):                                       │
│    validate(token)                                              │
│    extract_user(token)                                          │
│                                                                 │
│  class UserManager:                                             │
│    def __init__(self):                                          │
│      self.sessions = {}                                         │
│      ...more code...                                            │
└─────────────────────────────────────────────────────────────────┘

AST-BASED CHUNKING:
┌─────────────────────────────────────┐  ┌───────────────────────┐
│ Chunk 1: login() function           │  │ Chunk 2: authenticate()│
│ def login(user):                    │  │ def authenticate(token)│
│   verify(user)                      │  │   validate(token)     │
│   create_session(user)              │  │   extract_user(token) │
│   return token                      │  │                       │
└─────────────────────────────────────┘  └───────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Chunk 3: UserManager class + __init__                           │
│ class UserManager:                                              │
│   def __init__(self):                                           │
│     self.sessions = {}                                          │
│     ...more code...                                             │
└─────────────────────────────────────────────────────────────────┘

Pros: Complete functions/classes, semantic structure, better context
Cons: Larger chunks, potentially fewer chunks overall
```

### Usage Examples

```bash
# Default: AST-based chunks
python 2_lexical_rag_using_bm25_ast.py --repo ./my_project

# Custom repository
python 2_lexical_rag_using_bm25_ast.py --repo ~/projects/frontend

# Interactive session:
# You: What does the UserManager class do?
# You: Find all database query functions
# You: How is error handling implemented?
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `load_python_codebase()` | Loads all .py files as LangChain Documents |
| `chunk_code_ast()` | Parses AST, extracts classes/functions, creates semantic chunks |
| `build_bm25_retriever()` | Tokenizes AST chunks, creates BM25 index |
| `build_agent()` | Creates agent with retriever tool, middleware, and enhanced prompts |

### AST Advantages Over Recursive Splitting

```
Query: "What methods does UserManager have?"

File 1 (Recursive):
- Query returns arbitrary character-split chunks
- May include half of __init__ and half of another method
- Hard to understand full methods

File 2 (AST):
- Query returns complete class definition with all methods visible
- Each method is a separate, intact chunk
- Agent sees full context and structure
```

---

## Comparison: File 1 vs File 2

| Feature | File 1 (Recursive) | File 2 (AST) |
|---------|-------------------|-------------|
| **Chunking Method** | Fixed 1200-char splits | AST-based extraction |
| **Chunk Semantics** | Arbitrary (may split functions) | Complete (full functions/classes) |
| **Number of Chunks** | More (smaller) | Fewer (larger, complete) |
| **Chunk Coherence** | Low (mid-function cuts) | High (complete units) |
| **Setup Time** | Faster | Slower (AST parsing) |
| **Memory Usage** | Lower | Slightly higher |
| **Best For** | Quick proof-of-concept | Production use |
| **Retrieval Speed** | Same (both BM25) | Same (both BM25) |
| **Context Quality** | Moderate | Better |
| **Scalability** | Good | Better (fewer chunks) |

---

## Configuration

### Common Settings

```python
# File 1: Recursive chunking
CHUNK_SIZE = 1200  # Characters per chunk
                   # Smaller (500): More chunks, finer granularity, slower
                   # Larger (2000): Fewer chunks, broader context, less precision
                   # Common: 500, 1000, 1200, 1500

# File 2: AST chunking (no configuration needed)
# Chunks are determined by code structure, not size

# Both files use
LLM = "llama-3.1-8b-instant"  # Via Groq
temperature = 0  # Deterministic responses

# Safety middleware
ModelCallLimitMiddleware(run_limit=3)       # Max 3 LLM calls per query
ToolCallLimitMiddleware(run_limit=3)        # Max 3 retriever calls per query
```

### BM25 Parameters (from rank_bm25)

```python
BM25Okapi(
    corpus,
    k1=1.5,        # Term frequency saturation (higher = TF matters more)
    b=0.75,        # Length normalization (0 = no norm, 1 = full norm)
    epsilon=0.25   # IDF floor to prevent negative scores
)
```

For this RAG system, defaults are used and typically work well.

---

## When to Use Which File

### Use File 1 (Recursive Splitting) When:

- **Prototyping**: Quick setup, minimal configuration
- **Small codebases** (<100 files): Fast enough, simpler to understand
- **Exploring**: Testing what BM25 lexical search can do
- **Learning**: Understanding BM25 before moving to AST
- **Memory constrained**: Simpler chunks use less memory
- **Questions are specific**: Looking for exact function/variable names

**Example Use Cases:**
```
"Find the login function"
"Where is password_hash used?"
"Show me import statements"
"Find all error handling"
```

### Use File 2 (AST Splitting) When:

- **Production code search**: More reliable chunking
- **Large codebases** (100+ files): AST chunks scale better
- **Understanding structure**: Need to see complete functions/classes
- **Better context**: Questions about class methods, inheritance, structure
- **Reliable boundaries**: No worry about mid-function splits
- **Questions are structural**: Understanding relationships between components

**Example Use Cases:**
```
"What methods does the UserManager class have?"
"Show me the complete login flow"
"What does the DatabaseConnection class do?"
"How are decorators used in this codebase?"
```

---

## BM25 Search Examples

### Example 1: Exact Match
```
Query: "login"
Document 1: "def login(user): ..."
BM25 Score: High (exact term match)

Document 2: "def authenticate(token): ..."
BM25 Score: Low (no "login" term)
```

### Example 2: Multiple Terms (AND logic)
```
Query: "user authentication system"
Document 1: "def login(user): verify authentication"
BM25 Score: High (contains all terms: user, authentication, and implicit system context)

Document 2: "def get_user_name(): ..."
BM25 Score: Medium (has user, authentication, but not system context)
```

### Example 3: Term Frequency Matters
```
Query: "database connection"
Document 1: "def connect_db(): database.connect()" (mentioned once each)
BM25 Score: High

Document 2: "database database database..." (repeated "database", no "connection")
BM25 Score: Lower (TF normalization prevents spam scoring)
```

---

## Strengths and Limitations

### When BM25 Shines ✅
- Exact identifier lookups ("find UserManager class")
- Keyword-rich queries ("password hash salt")
- Fast retrieval (no neural network latency)
- Transparent results (see why matches occurred)
- Resource-efficient (no embeddings)
- Production-ready (proven, stable algorithm)

### When BM25 Struggles ❌
- Concept searches ("How do I implement caching?")
- Synonym handling ("login" vs "authenticate" vs "sign-in")
- Semantic similarity ("What's similar to this pattern?")
- Typos and misspellings
- Complex questions requiring reasoning
- Negation ("Find code that doesn't use X")

---

## Interactive Session Example

```
$ python 2_lexical_rag_using_bm25_ast.py --repo ./my_project
Loading codebase from: ./my_project
Loaded 42 Python files
Created 187 AST-based chunks
Building BM25 retriever...
BM25 retriever built successfully
Building agent...
Agent built successfully

============================================================
BM25 + AST Lexical RAG System Ready
============================================================
Ask your question about the codebase
Type 'exit' or 'quit' to end session

You: What is the UserManager class?
[Agent searches: finds UserManager class definition]
Agent: The UserManager class manages user sessions and authentication...

You: Show me the login flow
[Agent searches: finds login function, related functions]
Agent: The login flow calls verify_user(), create_session()...

You: exit
Exiting...
```

---

## Helper Dependencies

Both systems use utilities from `helpers/`:

| Function | Purpose |
|----------|---------|
| `load_python_codebase(path)` | Recursively loads all .py files as LangChain Documents |
| `chunk_code_recursive(docs, chunk_size)` | Character-based recursive splitting (File 1) |
| `chunk_code_ast(docs)` | AST-based extraction of functions/classes (File 2) |
| `BM25Retriever` | LangChain-compatible BM25 wrapper from rank_bm25 |

---

## Performance Characteristics

### File 1 (Recursive)
- **Indexing time**: ~100ms for 10K chunks
- **Query time**: ~10-50ms per search
- **Memory**: ~50MB for typical project
- **Scalability**: Good up to 50K chunks

### File 2 (AST)
- **Parsing time**: ~1-5 seconds (one-time, for all files)
- **Indexing time**: ~50ms for 500 chunks
- **Query time**: ~10-50ms per search
- **Memory**: ~30MB for typical project (fewer, larger chunks)
- **Scalability**: Excellent up to 500K chunks (real code, not microbenchmarks)

**Conclusion**: Both are fast. File 2 has slightly better memory efficiency and scalability due to fewer, coherent chunks.

---

## Troubleshooting

**"Could not find that in the codebase"**
- Try exact function/class names
- Check spelling and case sensitivity
- Use simpler queries (fewer terms)
- Try synonyms: "auth" vs "authenticate"

**Slow performance on first run**
- File 2 does AST parsing (one-time overhead)
- Subsequent queries are fast
- If still slow, check file count with `ls | wc -l`

**Irrelevant results**
- BM25 finds keyword matches, not semantic similarity
- Try more specific queries: "hash password" instead of "security"
- Use exact function names when known
- For semantic search, consider File 1 or 2 from `../3_graph_rag/` or `../4_hybrid_rag/`

**Memory issues**
- File 1: Reduce CHUNK_SIZE (e.g., 500 instead of 1200)
- File 2: Check for extremely large files (>10MB)
- Exclude large vendor directories (node_modules, .venv, etc.)
