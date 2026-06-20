# Triple Hybrid RAG Architecture

Visual representation of the triple hybrid RAG system and how all components work together.

## System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY INPUT                            │
│                                                                     │
│  "How does auth work and what depends on it?"                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIPLE HYBRID AGENT                              │
│  - Uses LangChain create_agent() with retriever tool                │
│  - Groq Llama 3.1 8B (temperature=0 for deterministic responses)    │
│  - Middleware: ModelCallLimitMiddleware(3), ToolCallLimitMiddleware │
│  - System prompt: Senior engineer with semantic+lexical+graph       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ↓ calls search_codebase tool
┌─────────────────────────────────────────────────────────────────────┐
│             TripleHybridRetriever (Custom Ensemble)                 │
│  - Orchestrates all three retrieval strategies                      │
│  - Merges results with deduplication                                │
│  - Returns top 8 merged results                                     │
└─────────┬──────────────────────────┬──────────────────────┬─────────┘
          │                          │                      │
          ↓ Semantic                 ↓ Lexical              ↓ Graph
┌─────────────────────────┐ ┌──────────────────────┐ ┌─────────────────┐
│  Semantic Retriever     │ │  Lexical Retriever   │ │ Graph Retriever │
│                         │ │                      │ │                 │
│ • HuggingFace embeddings│ │ • BM25 keyword index │ │ • Entity lookup │
│ • all-MiniLM-L6-v2      │ │ • Term frequency     │ │ • Ego traversal │
│ • Cosine similarity     │ │ • Fast keyword match │ │ • Depth=2 hops  │
│ • k=4 results           │ │ • k=4 results        │ │ • k=4 results   │
└────────┬────────────────┘ └─────────┬────────────┘ └───────┬─────────┘
         │                            │                      │
         ↓                            ↓                      ↓
    Vector Store              BM25 Okapi              Knowledge Graph
      (Chroma)                (in-memory)              (NetworkX)
         │                            │                      │
         │ Embedding Model            │ Tokenized chunks     │ Nodes: Entities
         │ ↓                          │                      │ Edges: IMPORTS
    all-MiniLM            [token, token, token]       auth → database
    (384 dims)               ↑       ↑       ↑        auth → permissions
                          tokenize split query        main → auth
                                                      (directed graph)
```

## Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│                  CONFIGURATION                          │
│  • CHUNK_SIZE = 500                                     │
│  • GRAPH_DEPTH = 2                                      │
│  • Default weights: 0.33/0.33/0.34                      │
└─────────────────────────────────────────────────────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │  Load & Chunk Codebase                 │
        │  ├─ load_python_codebase()             │
        │  └─ chunk_code_ast()                   │
        │     (extracts functions/classes)       │
        └────────────────────┬───────────────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │  Build Knowledge Graph                 │
        │  ├─ Nodes: entity names from chunks    │
        │  ├─ Edges: IMPORTS relationships       │
        │  └─ Graph: NetworkX DiGraph            │
        └────────────────────┬───────────────────┘
                             ↓
┌───────────────────┬────────────────────┬──────────────────────────┐
│                   │                    │                          │
↓                   ↓                    ↓                          ↓
Vector Store    BM25 Index          Graph Index            Metadata Map
(Chroma)        (rank_bm25)         (NetworkX)          (entity→chunks)
- Embeddings    - Tokenized         - Nodes & edges        - name: chunk
- Similarity    - Scored            - Relations            - source: path
- Store         - Fast lookup       - Traversable          - type: class/fn
```

## Code Organization

```
2_hybrid_rag_with_semantic_lexical_graph.py
│
├─ Imports & Setup
│  ├─ LangChain imports
│  ├─ NetworkX, Pydantic
│  └─ Helper functions import
│
├─ Section 1: Configuration
│  ├─ CHUNK_SIZE = 500
│  └─ GRAPH_DEPTH = 2
│
├─ Section 2: Pydantic Models
│  ├─ CodeRelationship (subject, predicate, obj)
│  ├─ GraphDocument (list of relationships)
│  └─ Entities (entity names from query)
│
├─ Section 3: Semantic Retriever
│  └─ build_vector_retriever()
│     ├─ Uses build_vector_store() from helpers
│     ├─ Returns retriever with k=4
│     └─ Uses HuggingFace embeddings
│
├─ Section 4: Lexical Retriever
│  └─ build_bm25_retriever_triple()
│     ├─ Uses BM25Retriever from helpers
│     ├─ Tokenizes chunks
│     └─ Returns BM25 retriever with k=4
│
├─ Section 5: Graph Retriever
│  ├─ GraphRetriever class
│  │  ├─ Attributes: graph, chunks_by_entity, k
│  │  ├─ _get_relevant_documents() - main retrieval logic
│  │  │  ├─ Extracts entities from query
│  │  │  ├─ Finds matching graph nodes
│  │  │  ├─ Gets ego graphs
│  │  │  └─ Returns associated chunks
│  │  └─ _extract_entities_simple() - heuristic entity extraction
│  └─ build_graph_retriever() - create instance
│
├─ Section 6: Triple Hybrid Ensemble
│  ├─ TripleHybridRetriever class
│  │  ├─ Attributes: semantic, lexical, graph retrievers + weights
│  │  └─ _get_relevant_documents()
│  │     ├─ Calls all three retrievers
│  │     ├─ Deduplicates by hash
│  │     ├─ Merges in priority order
│  │     └─ Returns top 8
│  └─ build_triple_hybrid_retriever() - create ensemble
│
├─ Section 7: Knowledge Graph Construction
│  └─ build_simple_graph_from_chunks()
│     ├─ Creates NetworkX DiGraph
│     ├─ Adds nodes from chunk metadata
│     ├─ Extracts IMPORTS from code
│     └─ Returns directed graph
│
├─ Section 8: Agent (25 lines)
│  └─ build_triple_hybrid_agent()
│     ├─ Creates retriever tool
│     ├─ Initializes Groq LLM
│     ├─ Creates agent with middleware
│     └─ Returns agent ready for use
│
└─ Section 9: Main
   ├─ Parse arguments (repo, weights)
   ├─ Validate weights sum to 1.0
   ├─ Load codebase
   ├─ Create AST chunks
   ├─ Build graph
   ├─ Build triple retriever
   ├─ Build agent
   └─ Interactive Q&A loop
```

## Retriever Specifications

### Semantic Retriever
```
Input: Query string
  ↓
Process: 
  1. Convert to 384-dim vector using all-MiniLM-L6-v2
  2. Search Chroma vector store
  3. Get k=4 most similar vectors
  4. Return corresponding Document chunks
  ↓
Output: List of 4 Documents (page_content + metadata)
```

### Lexical Retriever
```
Input: Query string
  ↓
Process:
  1. Tokenize into lowercase words
  2. Split on whitespace
  3. Score each chunk using BM25
  4. Get top k=4 by score
  5. Return corresponding Document chunks
  ↓
Output: List of 4 Documents (page_content + metadata)
```

### Graph Retriever
```
Input: Query string
  ↓
Process:
  1. Extract entities (capitalized words, identifiers)
  2. For each entity:
     a. Find matching nodes in graph
     b. Get ego graph at radius=2
     c. Extract edges and related entities
     d. Find chunks for those entities
  3. Deduplicate and limit to k=4
  ↓
Output: List of ≤4 Documents (page_content + metadata)
```

### Triple Ensemble
```
Input: Query string
  ↓
Process:
  1. Call Semantic Retriever → get 4 docs
  2. Call Lexical Retriever → get 4 docs
  3. Call Graph Retriever → get ≤4 docs
  4. Merge all results
     - Hash first 100 chars of each
     - Keep unique by hash
     - Preserve order: Semantic → Lexical → Graph
  5. Return top 8 documents
  ↓
Output: List of ≤8 Documents (merged from all three)
```

## Comparative Test
```bash
# Terminal 1: Semantic-heavy
python 2_hybrid_rag_with_semantic_lexical_graph.py --weights 0.6 0.2 0.2

# Terminal 2: Lexical-heavy
python 2_hybrid_rag_with_semantic_lexical_graph.py --weights 0.2 0.6 0.2

# Terminal 3: Graph-heavy
python 2_hybrid_rag_with_semantic_lexical_graph.py --weights 0.2 0.2 0.6

# Terminal 3: Equal-weight
python 2_hybrid_rag_with_semantic_lexical_graph.py --weights 0.34 0.33 0.33
python 2_hybrid_rag_with_semantic_lexical_graph.py --weights 0.4 0.3 0.3
```