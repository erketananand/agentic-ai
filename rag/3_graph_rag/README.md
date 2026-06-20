# Graph RAG Systems

This directory contains two complementary Graph RAG implementations for code analysis.

## System Comparison

| Feature | Chunking-Based Graph | LLM-Based Graph |
|---------|---|---|
| **File** | `1_graph_rag_using_chucking_graph_builder.py` | `2_graph_rag_using_llm_graph_builder.py` |
| **Graph Building** | Deterministic regex + heuristics | LLM extracts relationships |
| **Indexing Cost** | Free (fast) | Expensive (LLM per file) |
| **Relationship Types** | Fixed (4 types) | Rich & flexible |
| **Accuracy** | Good for standard patterns | Higher for complex patterns |
| **Speed** | Fast (~1-2 sec for 13 files) | Slower (LLM calls) |
| **Caching** | Fast caching (GraphML format) | Slower reuse |
| **Best For** | Fast iterations, high-frequency queries | Deep relationship analysis |

---

## Quick Start

### Chunking-Based Graph RAG (Recommended for Speed)
```bash
# Fast deterministic graph building, zero LLM indexing
python rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py

# With custom repository
python rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py --repo /path/to/code

# Force rebuild (skip cache)
python rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py --no-cache
```

### LLM-Based Graph RAG (For Rich Relationships)
```bash
# First run extracts relationships via LLM, then caches graph
python rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py

# Subsequent runs use cached graph
python rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py

# Adjust graph neighbourhood depth
python rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py --depth 3
```

---

# Chunking-Based Graph Builder Guide

## Overview

`rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py` implements a Graph RAG system that builds knowledge graphs **deterministically from code chunks** rather than using LLM extraction.

**Key advantage:** Zero LLM calls during indexing — only the Q&A agent uses Groq.

---

## How It Works

### 1. Code Loading & AST Chunking
```
load_python_codebase() → get all .py files
chunk_code_ast() → extract functions/classes with metadata
```
Each chunk has:
- `metadata['source']` — file path (e.g., `auth.py`)
- `metadata['name']` — function/class name (e.g., `authenticate`)
- `metadata['type']` — `"function"` or `"class"`

### 2. Graph Construction (4 Edge Types)

#### Phase 1: DEFINES Edges
```
module → entity
auth.py → authenticate
auth.py → check_permission
```
Created directly from chunk metadata (no LLM, no heuristics needed).

#### Phase 2: IMPORTS Edges
Scan each chunk for `import X` and `from X import` patterns:
```
database.py → models
notifications.py → email_client
```
Uses regex patterns:
- `^import\s+([\w.]+)` — direct imports
- `^from\s+([\w.]+)\s+import` — from imports

#### Phase 3: CALLS Edges
For each function/class, scan content for calls to known entity names:
```python
def create_user():
    # If check_permission() is called, add edge:
    # create_user → check_permission
```
Uses word-boundary regex: `\b{entity_name}\s*\(`

#### Phase 4: INHERITS_FROM Edges
For class chunks, detect base classes:
```python
class EmailService(NotificationBase):
    # If NotificationBase exists in graph:
    # EmailService → NotificationBase
```
Uses regex: `class\s+{name}\s*\(([\w, ]+)\)`

---

## Graph Retrieval

### Query → Graph Search → Chunks → LLM

```
User Query
    ↓
1. Token extraction (query.lower().split())
2. Match tokens against graph nodes (fuzzy matching)
3. Expand via ego graph (radius=2, undirected)
4. Collect chunks associated with matched nodes
5. Return top-4 chunks to agent
    ↓
Agent calls search_codebase tool
    ↓
LLM (Groq) answers using retrieved chunks
```

---

## Usage

### Basic Usage
```bash
cd C:\Users\kanand\Documents\ketan\agentic-ai
python rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py
```

### With Custom Repository
```bash
python rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py --repo /path/to/codebase
```

### Skip Cache (Rebuild Graph)
```bash
python rag/3_graph_rag/1_graph_rag_using_chucking_graph_builder.py --no-cache
```

### Example Session
```
Loading codebase from: C:\...\sample_project
Loaded 13 files
Chunking with AST...
Created 32 chunks
Building graph from chunks (DEFINES, IMPORTS, CALLS, INHERITS_FROM)...
Graph: 45 nodes, 67 edges
Graph saved to: rag/3_graph_rag/chunking_graph_cache.graphml
Building retriever and agent...

======================================================================
Graph RAG Ready. Type 'exit' or 'quit' to exit.

You: What functions does auth.py define?
Agent: auth.py defines several functions including...

You: How does the billing module use the database?
Agent: The billing module imports the database module...

You: exit
Exiting...
```

---

## Implementation Details

### Graph Nodes
- **Module nodes:** unique `metadata['source']` paths (e.g., `auth.py`)
- **Entity nodes:** `metadata['name']` from AST chunks (e.g., `authenticate`)
- Each node has attributes: `type` (module/function/class), `source` (file path)

### Graph Edges
All directed edges have a `relation` attribute (DEFINES, IMPORTS, CALLS, INHERITS_FROM).

### Retriever: GraphRetriever
Inherits from `BaseRetriever` (LangChain). Maps chunks by name for fast lookup:
```python
chunks_by_name = {
    "auth.py": [chunk1, chunk2, ...],
    "authenticate": [chunk3],
    "check_permission": [chunk4],
    ...
}
```

### Agent Tool
Wraps `GraphRetriever` as a `search_codebase` tool with Groq (Llama 3.1 8B).

---

## Edge Cases & Limitations

### Known Limitations
1. **IMPORTS detection:** Only finds top-level imports (inside functions not detected)
2. **CALLS detection:** Regex-based, may miss indirect calls or method chains
3. **INHERITS_FROM:** Only detects direct single inheritance (not multiple or metaclasses)
4. **Entity name conflicts:** If two functions have the same name in different files, they are treated as one node

### Workarounds
- For complex relationships, use LLM-based graph builder (file 2)
- For higher precision on specific queries, use Semantic or Lexical RAG
- Combine with `--no-cache` flag to force rebuild if codebase changes

---

---

# LLM-Based Graph Builder Guide

## Overview

`rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py` implements a Graph RAG system that uses **LLM to extract structured code relationships** and builds a rich knowledge graph.

**Key advantage:** Extracts complex relationships beyond standard patterns (DEFINES, IMPORTS, CALLS, INHERITS_FROM).

---

## How It Works

### 1. Code Loading
- Scans repository for Python files (excludes build artifacts, cache, VCS)
- Loads full content of each file

### 2. LLM Relationship Extraction
For each source file, the LLM agent:
- Analyzes code to identify relationships
- Extracts structured (subject, predicate, object) triples
- Returns as `GraphDocument` with list of `CodeRelationship` objects

**Relationship Types:**
- `DEFINES` — module defines a class/function
- `IMPORTS` — module imports another module
- `CALLS` — function calls another function
- `USES` — code uses a class/service
- `DEPENDS_ON` — module depends on another
- `INHERITS_FROM` — class inherits from base class
- `IMPLEMENTS` — class implements interface
- `SENDS_TO` — code sends data to external services
- `CONFIGURES` — code configures components

### 3. Graph Construction & Caching
- Converts relationship triples into NetworkX directed graph
- Saves graph in GraphML format for reuse
- Loads from cache on subsequent runs

### 4. Query Processing

```
User Query
    ↓
1. Entity extraction (LLM identifies classes, functions, modules)
2. Graph neighbourhood search (find matching nodes)
3. Expand ego graph (radius=2, undirected)
4. Extract and format relationships
5. Return knowledge graph context to QA agent
    ↓
LLM (Groq) answers with full context
```

---

## Usage

### Basic Usage
```bash
python rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py
```

### With Custom Repository
```bash
python rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py --repo /path/to/codebase
```

### Adjust Depth (Graph Neighbourhood)
```bash
python rag/3_graph_rag/2_graph_rag_using_llm_graph_builder.py --depth 3
```

### Example Session
```
Loading codebase from: C:\...\sample_project
Loaded 13 files

Extracting code relationships...
Extracted 87 relationships
Knowledge graph built: 45 nodes, 87 edges

KNOWLEDGE GRAPH - All extracted relationships:
  auth.py -[DEFINES]-> authenticate
  auth.py -[DEFINES]-> check_permission
  auth.py -[IMPORTS]-> models
  auth.py -[CALLS]-> database.query
  ...

======================================================================
Graph RAG System Ready
======================================================================

You: How does user creation work?
Agent: Based on the knowledge graph, user creation involves...
```

---

## Performance Notes

- **First run:** ~30-60 seconds (LLM extraction per file)
- **Cached runs:** <1 second (loads from GraphML)
- **Memory:** Graph stored as GraphML, easily cacheable
- **API Cost:** OpenAI calls for relationship extraction

---

## Configuration

Modify in the main block:
- **LLM Model for extraction:** `ChatOpenAI(model="gpt-4o-mini")`
- **LLM Model for QA:** `ChatGroq(model="llama-3.1-8b-instant")`
- **Graph neighbourhood depth:** `--depth` argument (default: 2)

---

## Shared Utilities

Both graph RAG systems use helpers from `helpers/graph_utils.py`:
- `save_graph()` — persist graph to GraphML
- `load_graph()` — load cached graph
- `match_nodes()` — fuzzy node matching for entity names
- `build_or_load_graph()` — unified caching with both graph types

---