# Chunking-Based Graph Builder Guide

## Overview

`rag/3_graph_rag/2_graph_rag_using_chucking_graph_builder.py` implements a Graph RAG system that builds knowledge graphs **deterministically from code chunks** rather than using LLM extraction.

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
python rag/3_graph_rag/2_graph_rag_using_chucking_graph_builder.py
```

### With Custom Repository
```bash
python rag/3_graph_rag/2_graph_rag_using_chucking_graph_builder.py --repo /path/to/codebase
```

### Skip Cache (Rebuild Graph)
```bash
python rag/3_graph_rag/2_graph_rag_using_chucking_graph_builder.py --no-cache
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

## Architecture Comparison

### vs. LLM-Based Graph RAG (File 1)
| Aspect | LLM-Based | Chunking-Based |
|---|---|---|
| Graph building cost | Expensive (LLM per file) | Free (regex + heuristics) |
| Edge types | Rich (extraction-dependent) | Deterministic (4 fixed types) |
| Accuracy for complex relationships | Higher | Good for standard patterns |
| Cache friendly | Slower (extract each time) | Fast (save after first build) |

### vs. Semantic/Lexical RAG
| Aspect | Chunking Graph | Semantic | Lexical |
|---|---|---|---|
| Relationship discovery | Native (graph traversal) | Limited | Limited |
| Multi-hop queries | Yes (up to radius-2) | No | No |
| Code location precision | Good | Fair | Excellent |
| Concept understanding | Fair | Excellent | Poor |

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
- For complex relationships, use LLM-based graph builder (file 1)
- For higher precision on specific queries, use Semantic or Lexical RAG
- Combine with `--no-cache` flag to force rebuild if codebase changes

---

## Files Modified/Created

| File | Action | Purpose |
|---|---|---|
| `helpers/graph_utils.py` | CREATE | Shared graph utilities (save/load/match) |
| `helpers/__init__.py` | UPDATE | Export graph utilities |
| `rag/3_graph_rag/2_graph_rag_using_chucking_graph_builder.py` | CREATE | Main implementation |

---

## Testing

Ask the system questions covering all retrieval strategies:

1. **DEFINES edge:** "What functions does the auth module define?"
2. **IMPORTS edge:** "What does the notifications module import?"
3. **CALLS edge:** "What functions does create_user call?"
4. **INHERITS_FROM edge:** "What base classes does EmailService inherit from?"
5. **Multi-hop:** "How does the task lifecycle work from creation to notification?"

---

## Performance Notes

- **Graph building:** ~1-2 seconds for 13 files (depends on code size)
- **Caching:** Saves ~1 second on subsequent runs (GraphML format, ~50KB)
- **Q&A latency:** ~2-3 seconds (1 retriever call + 1 LLM call, limited to 3 total)
- **Memory:** ~50MB for sample_project (chunking + graph + embeddings not used)

---

## Next Steps

1. Run with sample project to test basic functionality
2. Try with your own codebase: `--repo /your/codebase`
3. Compare answers with other RAG systems (semantic, lexical, LLM-based)
4. Adjust graph retrieval parameters if needed:
   - Change ego graph radius (default 2)
   - Adjust token filter length (default 2 chars)
   - Modify relationship detection regex patterns

---

## Troubleshooting

**No graph edges created:**
- Check that AST chunking is working: `print(chunks)` should show metadata
- Verify import statements are in chunks (print chunk content)
- Try `--no-cache` to force rebuild

**Retrieved chunks not relevant:**
- Query tokens may be too short or not matching node names
- Try longer entity names or file paths in your query
- Check graph nodes: `print(list(graph.nodes()))`

**Graph too small:**
- Ensure all functions/classes have `metadata['name']` (AST chunking prerequisite)
- Verify imports are correctly detected with `--no-cache`
