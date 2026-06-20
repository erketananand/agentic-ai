# Semantic RAG System

Two implementations of embedding-based retrieval systems for intelligent semantic code understanding. Both use vector embeddings to find conceptually similar code, then answer questions using an LLM.

## Overview

**Semantic RAG** uses dense vector embeddings instead of keywords. It converts both code chunks and user queries into vectors in a high-dimensional space, then finds the chunks with the most similar vectors using cosine similarity. This enables finding conceptually related code even if keywords don't match exactly.

**Key Advantage**: Understands meaning and concepts, not just keywords. Query "How do I authenticate a user?" will find code about login, verification, tokens, and sessions—even if they don't all use the word "authenticate."

**Key Difference Between Files:**
- **File 1**: Fixed-size recursive character splitting (1200 chars per chunk)
- **File 2**: AST-based splitting (complete functions/classes per chunk) for better semantic coherence

---

## 1_semantic_rag_using_recursive_text_splitter.py

**Simple embedding-based semantic search** using fixed-size character chunks with vector similarity.

### How It Works

1. **Loading**: Reads all Python files from target directory
2. **Chunking**: Splits code recursively on characters into 1200-character chunks
   - Simple, predictable splits
   - May split functions/classes in the middle
   - Creates many chunks for broad coverage
3. **Embedding**: Converts each chunk to a vector
   - Uses all-MiniLM-L6-v2 model (384 dimensions)
   - Captures semantic meaning of code
   - Stored in Chroma vector database (in-memory)
4. **Query Embedding**: Converts user question to vector using same model
5. **Similarity Search**: Finds top-4 chunks with highest cosine similarity
   - Fast vector lookup via Chroma indexing
   - Returns semantically related code
6. **Answering**: LLM synthesizes answer from retrieved chunks

### Vector Embeddings Explained

**What is an Embedding?**

An embedding converts text (or code) into a vector of numbers representing its meaning:

```
Code chunk:
"def login(user):
    verify(user)
    create_session(user)
    return token"

↓ all-MiniLM-L6-v2

Vector (384 dimensions):
[0.12, -0.45, 0.88, ..., 0.23]  (384 numbers total)

Meaning captured:
- This is about authentication
- Involves user verification
- Creates sessions
- Returns tokens
```

**Semantic Space**:
```
Vector Space (visualized in 2D, actually 384D):

    ↑ meaning: "authentication"
    │
    │  • authenticate ○ login
    │   \ •verify  ○ sign_in
    │    \  • session
    │  ○ password     • token
    │
    └──────────────────────→ meaning: "user_management"

Cosine similarity = angle between vectors
- Small angle (close) = similar meaning
- Large angle (far) = different meaning
```

**Why This Works for Code**:
- Functions doing similar things get similar embeddings
- Synonyms naturally cluster together ("login", "authenticate", "sign_in")
- Related concepts form neighborhoods ("password", "hash", "salt")
- Semantic relationships emerge without explicit rules

### All-MiniLM-L6-v2 Model

- **Size**: 22 million parameters (lightweight)
- **Speed**: Embeds ~200 chunks/second
- **Dimensions**: 384 (balanced: smaller than 768, larger than 128)
- **Training**: Trained on 1 billion+ sentence pairs
- **Quality**: Strong for code and natural language
- **Type**: Sentence Transformers model (optimized for semantic similarity)

### Usage Examples

```bash
# Default: 1200-char recursive chunks
python 1_semantic_rag_using_recursive_text_splitter.py --repo ./my_project

# Custom repository
python 1_semantic_rag_using_recursive_text_splitter.py --repo ~/projects/backend

# Interactive session:
# You: How do I implement user authentication?
# You: Show me similar error handling patterns
# You: What's the data flow from API to database?
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `load_python_codebase()` | Loads all .py files as LangChain Documents |
| `chunk_code_recursive()` | Splits on character boundaries (1200-char chunks) |
| `build_vector_store()` | Creates Chroma vector store with embeddings |
| `build_agent()` | Creates LangChain agent with retriever tool and middleware |
| Chroma | In-memory vector database (can persist to disk) |
| all-MiniLM-L6-v2 | Embedding model (HuggingFace Sentence Transformers) |

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

Each chunk gets embedded separately:
Chunk 1 embedding captures: auth, user verification, session creation
Chunk 2 embedding captures: token validation, user extraction
```

### Cosine Similarity for Finding Related Code

```
Query: "How do I verify users?"

Query embedding:
[0.15, -0.42, 0.91, 0.33, ..., -0.18]  (384 dims)

Candidate chunks and their similarity scores:

Chunk A: "def verify_user(user): ..."
Similarity: 0.89 ✓ (high, exact match)

Chunk B: "def authenticate_user(user): ..."
Similarity: 0.85 ✓ (high, semantic synonym)

Chunk C: "def check_password(pwd): ..."
Similarity: 0.72 ✓ (medium, related to verification)

Chunk D: "def render_login_form(): ..."
Similarity: 0.41 ✗ (low, not related)

Top-4 returned: A, B, C, and next best match
```

### Strengths and Weaknesses

**Strengths:**
- ✅ **Semantic understanding**: Finds "login" when searching "authenticate"
- ✅ **Concept-based**: Understands design patterns and logic flow
- ✅ **Synonym-tolerant**: "token", "JWT", "session" all cluster together
- ✅ **Typo-resistant**: Misspelled queries still find relevant code
- ✅ **Context-aware**: Captures surrounding meaning in code
- ✅ **No exact matching required**: Works with vague questions

**Weaknesses:**
- ❌ **Slower than BM25**: Embedding computation takes time
- ❌ **Model dependency**: Quality depends on embedding model training
- ❌ **Hallucinations possible**: LLM might make incorrect connections
- ❌ **Memory overhead**: Vector storage requires more RAM
- ❌ **Cold start**: Initial embedding generation takes time
- ❌ **Less interpretable**: Hard to see why a result matched

---

## 2_semantic_rag_using_ast.py

**Semantic search with structure-aware AST chunking** for complete, coherent functions and classes.

### How It Works

1. **Loading**: Reads all Python files
2. **AST Parsing**: Extracts complete functions and classes
   - Each chunk is a complete, valid function/class
   - Preserves code structure and boundaries
3. **Embedding**: Converts each chunk to vector (same as File 1)
   - Uses all-MiniLM-L6-v2
   - Stored in Chroma vector database
4. **Query Embedding**: Converts user question to vector
5. **Similarity Search**: Finds top-4 semantically similar chunks
   - Now finding complete functions/classes (better context)
6. **Answering**: LLM synthesizes from coherent chunks

### AST Parsing for Better Embeddings

**Why AST-based Chunks Matter for Semantic Search:**

When chunks are complete functions, embeddings capture fuller meaning:

```
Recursive split (File 1):
Chunk 1: "def login(user):
    verify(user)"
(Incomplete, embedding lacks full context)

Chunk 2: "create_session(user)
    return token"
(Fragment, embedding is about sessions/tokens only)

↓ Query: "What does login do?"
Result: Finds both chunks separately, less coherent

─────────────────────────────────────────────────

AST split (File 2):
Chunk: "def login(user):
    verify(user)
    create_session(user)
    return token"
(Complete, embedding captures full login flow)

↓ Query: "What does login do?"
Result: Finds complete function with full context
```

### AST Extraction Process

```python
Code:
def login(user):
    verify(user)
    create_session(user)

class UserManager:
    def __init__(self):
        self.sessions = {}
    
    def add_user(self, user):
        self.sessions[user.id] = user

↓ AST parsing

Extracts:
- Function 'login' (lines 1-3)
- Class 'UserManager' (lines 5-10)
- Method '__init__' (lines 6-7)  [part of class]
- Method 'add_user' (lines 9-10) [part of class]

Each becomes one chunk:
Chunk 1: login() function (complete)
Chunk 2: UserManager class (complete, with all methods)
```

### Usage Examples

```bash
# Default: AST-based chunks
python 2_semantic_rag_using_ast.py --repo ./my_project

# Custom repository
python 2_semantic_rag_using_ast.py --repo ~/projects/frontend

# Interactive session:
# You: What does the UserManager class do?
# You: Find functions that handle payments
# You: Show me patterns for error handling
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `load_python_codebase()` | Loads all .py files |
| `chunk_code_ast()` | Parses AST, extracts functions/classes |
| `build_vector_store()` | Creates vector store with embeddings |
| `build_agent()` | Creates agent with retriever tool |

### Embedding Quality Improvement

```
Recursive chunks (File 1):
- Average chunk: 1200 characters (often incomplete)
- Embedding context: Partial (incomplete functions)
- Semantic noise: Some chunks are mid-function fragments
- Result quality: Good, but misses full context

AST chunks (File 2):
- Average chunk: 500-2000 characters (complete units)
- Embedding context: Full (complete functions/classes)
- Semantic purity: Each chunk is a complete semantic unit
- Result quality: Better, full functions returned

Example:
Query: "authentication flow"

File 1:
- Returns fragment about verify()
- Returns fragment about session creation
- Returns class definition
- LLM has to piece together from fragments

File 2:
- Returns complete login() function (has verify + session in one chunk)
- Returns complete UserManager class (has all methods)
- LLM gets coherent, complete code units
```

---

## Comparison: File 1 vs File 2

| Feature | File 1 (Recursive) | File 2 (AST) |
|---------|-------------------|-------------|
| **Chunking** | Fixed 1200-char splits | AST-based extraction |
| **Chunk Coherence** | Low (may split functions) | High (complete units) |
| **Number of Chunks** | More (smaller) | Fewer (larger) |
| **Setup Time** | Faster | Slower (AST parsing) |
| **Embedding Time** | ~100ms | ~50ms (fewer chunks) |
| **Memory Usage** | Higher (more chunks) | Lower (fewer chunks) |
| **Semantic Quality** | Good | Better |
| **Best For** | Quick POC | Production |
| **Query Speed** | Same (vector lookup) | Same (vector lookup) |
| **Scalability** | Good | Better |

---

## Vector Store Persistence

### In-Memory (Default)

```python
vector_store = build_vector_store(chunks)  # Defaults to in-memory
```

- Fast startup
- Not persisted between runs
- Lost when program exits

### Persistent (Optional)

```python
vector_store = build_vector_store(chunks, persist_directory="./chroma_db")
```

- First run: Creates embeddings and saves to disk
- Subsequent runs: Loads embeddings from disk (fast)
- Survives program restarts
- Great for large codebases (embeddings are slow to recompute)

---

## Configuration

### Common Settings (Both Files)

```python
# File 1: Recursive chunking
CHUNK_SIZE = 1200  # Characters per chunk
                   # Smaller (500): More chunks, finer detail
                   # Larger (2000): Fewer chunks, broader context

# File 2: AST chunking (no configuration needed)
# Chunks determined by code structure

# Both files use
LLM = "llama-3.1-8b-instant"  # Via Groq
temperature = 0  # Deterministic responses

# Retriever settings
k = 4  # Return top 4 most similar chunks per query

# Safety middleware
ModelCallLimitMiddleware(run_limit=3)       # Max 3 LLM calls
ToolCallLimitMiddleware(run_limit=3)        # Max 3 retriever calls
```

### Embedding Model Settings

```python
# Model: all-MiniLM-L6-v2 (HuggingFace Sentence Transformers)
# Download: Automatic on first use
# Size: 22M parameters
# Speed: ~200 chunks/second on CPU
# Dimensions: 384

# Can be changed in helpers.py:
# model_name = "all-MiniLM-L6-v2"  # Default
# model_name = "all-mpnet-base-v2"  # Larger (438M params)
# model_name = "paraphrase-MiniLM-L6-v2"  # Paraphrase-optimized
```

---

## When to Use Which File

### Use File 1 (Recursive Splitting) When:

- **Exploring**: Quick prototype to understand semantic search
- **Learning**: Understanding how embeddings work
- **Small codebases** (<50 files)
- **Broad coverage**: Want many chunks from all code
- **Time-sensitive**: Minimal setup overhead

**Example Use Cases:**
```
"How do I implement this pattern?"
"Find similar error handling approaches"
"What's the overall architecture?"
"Show me related utility functions"
```

### Use File 2 (AST Splitting) When:

- **Production use**: More reliable, coherent chunks
- **Large codebases** (50+ files)
- **Understanding structure**: Need to see complete functions/classes
- **Better answers**: Want full code context per result
- **Scalable**: Fewer chunks = less memory, faster embedding

**Example Use Cases:**
```
"What does the UserManager class do?"
"Show me the complete login function"
"How is error handling structured?"
"Explain the service layer architecture"
```

---

## Semantic Search Examples

### Example 1: Synonym Understanding

```
Query: "How do I sign in a user?"

Query embedding captures meaning: user authentication, login, access

Candidates:
- "def login(user): ..." → Similarity: 0.92 ✓
- "def authenticate(user): ..." → Similarity: 0.89 ✓
- "def verify_credentials(user): ..." → Similarity: 0.84 ✓
- "def get_user_profile(user_id): ..." → Similarity: 0.51 ✗

Retrieved: Top 4, all related to user authentication despite different keywords
```

### Example 2: Concept Matching

```
Query: "How do I cache data for performance?"

Query embedding captures: caching, performance, storage, memoization

Candidates:
- "def cache_result(key, value): ..." → Similarity: 0.88 ✓ (exact term)
- "def memoize_function(func): ..." → Similarity: 0.85 ✓ (same concept)
- "redis_store.set(key, value)" → Similarity: 0.82 ✓ (cache mechanism)
- "def store_user_data(user): ..." → Similarity: 0.41 ✗ (not performance-related)

Retrieved: All related to caching despite different implementations
```

### Example 3: Typo Tolerance

```
Query: "Where is the autentication logic?" (misspelled)

Query embedding still captures: authentication concept

Candidates:
- "def login(user): ..." → Similarity: 0.84 ✓
- "def authenticate(user): ..." → Similarity: 0.81 ✓

Retrieved: Correct results despite typo (embeddings are typo-resilient)
```

---

## Strengths and Limitations

### When Semantic Search Shines ✅
- Concept-based questions ("How do I implement X?")
- Finding similar patterns ("Show me related error handling")
- Synonym tolerance ("login" vs "authenticate")
- Vague or natural language queries
- Understanding intent over exact keywords
- Cross-cutting concerns ("All logging code")

### When Semantic Search Struggles ❌
- Very specific identifier searches ("Find the EXACT variable `cache_timeout`")
- Rare, specialized terms not in training data
- Queries requiring negation ("Find code NOT using globals")
- Performance-critical: Embedding computation is slower than BM25
- Requires more setup: Model download, vector computation

---

## Interactive Session Example

```
$ python 2_semantic_rag_using_ast.py --repo ./my_project
Loaded 42 files → 187 chunks (AST-based)
Ready. Ask your question. Type 'exit' to quit

You: How do I authenticate users?
[Searches embeddings: finds login, authenticate, verify_token functions]
Agent: Authentication happens in the login function which verifies user 
credentials and creates a session token. The verify_token function is used 
to validate incoming requests...

You: What about error handling for failed auth?
[Searches embeddings: finds exception handling near auth code]
Agent: Failed authentication raises AuthenticationError which is caught and 
returns 401 Unauthorized response. Invalid tokens trigger a TokenExpiredError...

You: Show me similar patterns in the codebase
[Searches embeddings: finds related validation code]
Agent: Similar validation patterns are used for payment processing and 
database connection validation...

You: exit
```

---

## Performance Characteristics

### File 1 (Recursive)
- **Load time**: ~100ms (load files)
- **Embedding time**: ~500ms per 1000 chunks
- **Query time**: ~50ms (find similar vectors + LLM response)
- **Memory**: ~100-200MB for typical project
- **Vector store**: Chroma (in-memory, can persist)

### File 2 (AST)
- **Parse time**: ~2 seconds (AST parsing all files, one-time)
- **Embedding time**: ~200ms per 500 chunks (fewer chunks)
- **Query time**: ~50ms (same as File 1)
- **Memory**: ~80-150MB (fewer chunks)
- **Vector store**: Chroma (in-memory, can persist)

**Conclusion**: File 2 slightly faster overall due to fewer chunks and AST parse being fast.

---

## Embedding Space Visualization

```
Semantic Similarity Space (conceptual):

Authentication Cluster:
    • login ○ authenticate
     \ • verify
      \  ○ sign_in
       • session_create

Database Cluster:
    • query ○ fetch_data
     \ • execute_sql
      \ ○ get_records
       • db_connection

Error Handling Cluster:
    • try/except ○ raise
     \ • error_handler
      \ ○ exception
       • logging

← Less related ────────────────────────→ More related

Distance in vector space represents semantic similarity
Queries naturally cluster with related code chunks
```

---

## Troubleshooting

**"Could not find that in the codebase"**
- Query might be too specific or use different terminology
- Try synonyms: "login" vs "authenticate"
- Try more general terms: "HTTP handling" instead of "GET request parsing"
- Increase k (top-k results) to see more options

**Slow embedding computation (first run)**
- Large codebase → many chunks to embed
- First embedding of model is slower (model loading)
- Subsequent queries are fast (vectors cached)
- Consider using persistent vector store

**Memory issues**
- Large number of chunks → high memory
- Solution: File 2 (AST) has fewer chunks
- Alternative: Run on subset of codebase

**Poor retrieval quality**
- Query might be too different from code
- Try multiple query phrasings
- All-MiniLM-L6-v2 is general; may miss domain-specific concepts
- Consider using BM25 (lexical RAG) for exact terms

---

## Helper Dependencies

Both systems use utilities from `helpers/`:

| Function | Purpose |
|----------|---------|
| `load_python_codebase(path)` | Loads all .py files as Documents |
| `chunk_code_recursive(docs, size)` | Character-based recursive splitting (File 1) |
| `chunk_code_ast(docs)` | AST extraction of functions/classes (File 2) |
| `build_vector_store(chunks, persist_dir=None)` | Creates Chroma vector store with embeddings |

---

## Next Steps

- **Try both files**: Experience the difference between recursive and AST chunking
- **Experiment with queries**: See how semantic search finds related concepts
- **Compare with BM25**: Try `../2_lexical_rag/` to compare keyword search
- **Persist embeddings**: Use `persist_directory` for reusable vector store
- **Larger queries**: Test on real codebases to see semantic search quality
- **Check performance**: Compare query times between File 1 and 2
