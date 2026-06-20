# Hybrid RAG System

Two implementations of hybrid retrieval systems that combine semantic and lexical search strategies for intelligent code understanding and question-answering over Python codebases.

## Overview

Both systems use LangChain agents with custom retrievers to find relevant code snippets, then answer questions using an LLM (Llama 3.1 8B via Groq). The key difference: File 1 uses two retrieval strategies (semantic + lexical), while File 2 adds a third strategy (knowledge graph) for relationship-aware retrieval.

---

## 1_hybrid_rag_with_semantic_lexical.py

**Dual-retriever hybrid system** combining vector embeddings with keyword matching for comprehensive code search.

### How It Works

1. **Chunking**: Splits Python codebase into 500-character chunks
2. **Dual Retrieval**:
   - **Semantic Retriever**: Converts chunks to embeddings (HuggingFace all-MiniLM-L6-v2), finds 4 most similar vectors using cosine similarity
   - **Lexical Retriever**: Uses BM25 (probabilistic ranking function), tokenizes chunks, finds 4 best keyword matches
3. **Merging**: EnsembleRetriever combines results using Reciprocal Rank Fusion (RRF) with adjustable weights
4. **Agent**: AI agent uses the merged results to answer questions, references specific files/functions in responses

### Retrievers Explained

**Semantic Retriever**
- Uses all-MiniLM-L6-v2 model (384-dimensional embeddings)
- Finds conceptually similar code (design patterns, implementations, logic flow)
- Best for: "How do I implement authentication?" or "Find similar error handling patterns"
- Returns: Top 4 most semantically similar chunks

**Lexical Retriever**
- Uses BM25 (Okapi algorithm) for probabilistic keyword scoring
- Finds exact matches for identifiers, function names, class names
- Best for: "Where is the login function?" or "Find all uses of database_config"
- Returns: Top 4 chunks with highest keyword match scores

**Ensemble Merger (RRF)**
- Reciprocal Rank Fusion: Combines rankings from both retrievers
- Formula: 1/(k + rank) for each result, normalized by weight
- Deduplicates results, preserves unique findings from both strategies
- Default weights: 0.5 semantic + 0.5 lexical (equal blending)

### Usage Examples

```bash
# Default: equal blending (0.5 semantic, 0.5 lexical)
python 1_hybrid_rag_with_semantic_lexical.py --repo ./my_project

# Favor semantic search (70% semantic, 30% lexical)
# Better for concept-based questions about architecture/patterns
python 1_hybrid_rag_with_semantic_lexical.py --repo ./my_project --weights 0.7 0.3

# Favor lexical search (30% semantic, 70% lexical)
# Better for specific function/variable lookups
python 1_hybrid_rag_with_semantic_lexical.py --repo ./my_project --weights 0.3 0.7
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `build_vector_retriever()` | Creates semantic retriever with HuggingFace embeddings, returns k=4 results |
| `build_bm25_retriever_hybrid()` | Creates lexical retriever using BM25, tokenizes all chunks, returns k=4 results |
| `build_hybrid_retriever()` | Combines both retrievers into EnsembleRetriever with RRF merging |
| `build_agent()` | Creates LangChain agent with retriever tool, adds middleware for safety (max 3 LLM calls, max 3 tool calls) |

---

## 2_hybrid_rag_with_semantic_lexical_graph.py

**Triple-retriever hybrid system** that adds knowledge graph analysis to understand code relationships and architecture.

### How It Works

1. **Chunking**: Splits code using AST (Abstract Syntax Tree) for better semantic structure, creates 500-character chunks
2. **Graph Construction**: Builds NetworkX directed graph with:
   - Nodes: Code entities (functions, classes, modules)
   - Edges: Relationships (IMPORTS, USES, CALLS, DEPENDS_ON)
   - Caches to `hybrid_graph_cache.graphml` for reuse
3. **Triple Retrieval**:
   - **Semantic**: Vector embeddings (same as File 1)
   - **Lexical**: BM25 keyword matching (same as File 1)
   - **Graph**: Entity lookup + ego-graph traversal (depth=2)
4. **Merging**: TripleHybridRetriever combines all three with deduplication
5. **Agent**: Enhanced agent understands relationships between modules and explains architecture

### Retrievers Explained

**Semantic Retriever** (identical to File 1)
- Vector embeddings for conceptual similarity
- k=4 results

**Lexical Retriever** (identical to File 1)
- BM25 keyword matching
- k=4 results

**Graph Retriever** (new to File 2)
- Extracts entity keywords from query (capitalized words, identifiers)
- For each entity:
  - Finds matching nodes in the knowledge graph
  - Gets ego-graph (2-hop neighborhood: entity + direct neighbors + neighbors of neighbors)
  - Extracts all relationships within that subgraph
  - Retrieves code chunks associated with those entities
- Returns: ≤4 chunks related to graph entities
- Best for: "What depends on the auth module?" or "Show me the dependency chain from main to database"

**Triple Ensemble**
- Calls all three retrievers independently
- Deduplicates results by hashing first 100 characters (prevents duplicate context)
- Merges in priority order: Semantic → Lexical → Graph
- Returns: Top 8 documents (unique results from all three strategies)

### Usage Examples

```bash
# Default: equal weights (0.33 semantic, 0.33 lexical, 0.34 graph)
python 2_hybrid_rag_with_semantic_lexical_graph.py --repo ./my_project

# Semantic-focused (60% semantic, 20% lexical, 20% graph)
# Better for pattern recognition and concept understanding
python 2_hybrid_rag_with_semantic_lexical_graph.py --repo ./my_project --weights 0.6 0.2 0.2

# Graph-focused (20% semantic, 20% lexical, 60% graph)
# Better for understanding architecture, dependencies, data flow
python 2_hybrid_rag_with_semantic_lexical_graph.py --repo ./my_project --weights 0.2 0.2 0.6

# Balanced with slightly more graph (40% semantic, 30% lexical, 30% graph)
python 2_hybrid_rag_with_semantic_lexical_graph.py --repo ./my_project --weights 0.4 0.3 0.3
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `build_vector_retriever()` | Semantic search with embeddings, k=4 |
| `build_bm25_retriever_triple()` | Lexical search with BM25, k=4 |
| `GraphRetriever` | Custom retriever class that performs entity extraction and ego-graph traversal |
| `_extract_entities_simple()` | Heuristic extraction: capitalized words, identifiers with underscores, valid Python identifiers |
| `_get_relevant_documents()` | Main logic: find matching nodes, traverse ego-graph, collect related chunks |
| `TripleHybridRetriever` | Ensemble orchestrator combining all three retrievers with deduplication |
| `build_triple_hybrid_agent()` | Creates agent with enhanced system prompt referencing architecture and relationships |

### Knowledge Graph Details

**Graph Construction**:
- Nodes extracted from chunk metadata (function names, class names, module paths)
- Edges inferred from code analysis (import statements, function calls, class inheritance)
- Stored as NetworkX DiGraph (directed, supports multi-hop traversal)

**Graph Retrieval Process**:
- Query "What calls the auth_manager?" → Extracts entity "auth_manager"
- Finds matching node in graph: `auth_manager` (exact match)
- Gets ego-graph with radius=2 (all nodes connected within 2 hops)
- Returns chunks for those entities

**Caching**:
- Graph saved to `hybrid_graph_cache.graphml` after first build
- Subsequent runs load from cache (much faster)
- Pass `--no-cache` equivalent if you need to rebuild (default uses cache)

---

## Configuration

### Common Settings (Both Files)

```python
CHUNK_SIZE = 500  # Characters per chunk
                  # Smaller = finer granularity (more chunks, slower)
                  # Larger = coarser granularity (fewer chunks, less precision)
                  # Try: 250, 500, 1000, 1500

# Embedding model
MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, good quality

# LLM settings
LLM = "llama-3.1-8b-instant"  # Via Groq (fast, free tier available)
temperature = 0  # Deterministic responses (0 = always same output)

# Safety middleware
ModelCallLimitMiddleware(run_limit=3)       # Max 3 LLM calls per query
ToolCallLimitMiddleware(run_limit=3)        # Max 3 retriever calls per query
```

### File 2 Additions

```python
GRAPH_DEPTH = 2  # Hops for ego-graph traversal
                 # 1 = direct neighbors only
                 # 2 = neighbors of neighbors (recommended)
                 # 3+ = broader context, slower

GRAPH_CACHE_PATH = Path(__file__).parent / "hybrid_graph_cache.graphml"
# Persistent storage for knowledge graph (reused across runs)
```

---

## Helper Dependencies

Both systems rely on utilities from `helpers/` directory:

| Function | Purpose |
|----------|---------|
| `load_python_codebase(path)` | Recursively loads all .py files as LangChain Documents |
| `chunk_code_recursive(docs, chunk_size)` | Splits code on character boundaries (File 1) |
| `chunk_code_ast(docs)` | Splits code at AST boundaries: functions, classes (File 2, better structure) |
| `build_vector_store(chunks)` | Creates Chroma vector store with embeddings |
| `BM25Retriever` | LangChain-compatible wrapper around rank_bm25 |
| `build_or_load_graph(chunks, cache_path)` | Builds NetworkX graph or loads from cache (File 2 only) |
