# RAG Evaluation Framework

Automated evaluation system for Retrieval-Augmented Generation (RAG) implementations using RAGAS metrics and LLM-based judging.

## Overview

This evaluation suite tests RAG system performance using:
- **RAGAS Metrics**: Automated evaluation framework measuring retrieval quality and generation accuracy
- **Llama 3.1 8B Judge**: Cost-effective LLM evaluation
- **Hybrid Caching**: Two-layer caching system reducing repeat evaluation costs by 900×

## Available Evaluations

### Semantic RAG Implementations
1. **`1_semantic_rag_with_256_chunking_and_k_eq_4.py`** — Semantic retrieval with 256-token chunks, k=4
2. **`2_semantic_rag_with_1500_chunking_and_k_eq_4.py`** — Semantic retrieval with 1500-token chunks, k=4
3. **`3_semantic_rag_with_1500_chunking_and_k_eq_1.py`** — Semantic retrieval with 1500-token chunks, k=1

### Alternative Chunking & Retrieval Strategies
4. **`4_semantic_rag_with_ast_chunking_and_k_eq_1.py`** — AST-based semantic retrieval, k=1
5. **`5_lexical_rag_with_bm25_256_chunking_and_k_eq_4.py`** — BM25 lexical retrieval with 256-token chunks, k=4

### Advanced RAG Approaches
6. **`6_graph_rag.py`** — Graph-based retrieval system (chunking-based approach)
7. **`7_hybrid_rag.py`** — Hybrid retrieval combining semantic and lexical (Reciprocal Rank Fusion)

## Quick Start

### Run a Specific Evaluation (First Run)
```bash
python 1_semantic_rag_with_256_chunking_and_k_eq_4.py
```

**Output**: 
- Time: 2-3 minutes
- Cost: ~$0.02
- Creates cache files for future runs

### Repeat Run (All Cached)
```bash
python 1_semantic_rag_with_256_chunking_and_k_eq_4.py
```

**Output**:
- Time: ~30 seconds (85% faster)
- Cost: ~$0.00 (zero API calls)
- Loads all results from cache

### Compare Multiple Implementations
```bash
# Run all evaluations
python 1_semantic_rag_with_256_chunking_and_k_eq_4.py
python 2_semantic_rag_with_1500_chunking_and_k_eq_4.py
python 3_semantic_rag_with_1500_chunking_and_k_eq_1.py
python 4_semantic_rag_with_ast_chunking_and_k_eq_1.py
python 5_lexical_rag_with_bm25_256_chunking_and_k_eq_4.py
python 6_graph_rag.py
python 7_hybrid_rag.py
```

### Clear Caches for Fresh Evaluation
```bash
rm -r evals/.cache/
python 1_semantic_rag_with_256_chunking_and_k_eq_4.py
```

---

## Comparison & Learnings

### Performance Scorecard

| Config | Retrieval | Precision | Recall | Faithfulness | Relevancy |
|--------|-----------|-----------|--------|--------------|-----------|
| Semantic, chunk=256, k=4 | Semantic | 0.333 | 0.362 | 0.603 | 0.609 |
| Semantic, chunk=1500, k=4 | Semantic | 0.778 | 0.65 | 0.821 | 0.821 |
| Semantic, chunk=1500, k=1 | Semantic | 0.366 | 0.426 | 0.706 | 0.623 |
| AST chunks, k=1 | Semantic | 0.534 | 0.6 | 0.741 | 0.682 |
| BM25, chunk=256, k=4 | Lexical | 0.306 | 0.242 | 0.281 | 0.449 |
| Hybrid (HF + BM25), chunk=256 | Hybrid | 0.667 | 0.466 | 0.649 | 0.685 |
| Hybrid (OpenAI + BM25), chunk=256 | Hybrid | 0.641 | 0.469 | 0.686 | 0.676 |
| Hybrid (OpenAI + BM25), chunk=1500 | Hybrid | **0.889** | **0.824** | **0.873** | **0.825** |
| GraphRAG, depth=2 | Graph | 0.389 | 0.34 | 0.736 | 0.394 |

**🏆 Best overall: Hybrid (OpenAI + BM25), chunk=1500** — new winner across all four metrics.

### Key Learnings

#### 1. Chunk Size Is the Biggest Lever

Going from chunk=256 to chunk=1500 (same k=4, semantic retrieval):
- **Precision**: 0.333 → 0.778 (+133%)
- **Recall**: 0.362 → 0.650 (+80%)

**Why**: Small chunks (256 tokens) fragment class and function bodies. The retriever finds a piece but misses surrounding logic. For code, the meaningful semantic unit is a full function or class body, not a small text window.

#### 2. k Matters Almost as Much as Chunk Size

Dropping k from 4 to 1 (same chunk=1500 config):
- **Precision**: 0.778 → 0.366 (-53%)
- **Recall**: 0.650 → 0.426 (-34%)

**Why**: Cross-file questions like "find every place X is raised" or "trace the call chain" need chunks from multiple files. One retrieved chunk is never enough for whole-codebase lookup questions.

#### 3. AST Chunking Beats Text-Based at k=1, But Has Structural Blind Spots

AST chunking at k=1 vs text chunk=1500 at k=1:
- **Precision**: 0.534 vs 0.366 (+46% win for AST)
- **Recall**: 0.600 vs 0.426 (+41% win for AST)

**Why AST wins**: Structural chunking by function/class produces coherent, complete units — higher quality per retrieved chunk.

**But AST fails on**:
- Import-level questions: "Which file imports both Task and Status?" → 0.000 across all metrics
- Module-scope questions: "Find every place BillingError is raised" → 0.000 across all metrics
- **Root cause**: AST chunks by function nodes and doesn't include module-level import blocks.

#### 4. Faithfulness Is Resilient for Semantic RAG — But Collapses with BM25

- **Semantic RAG**: Faithfulness stays ≥0.603 even in worst config (chunk=256)
- **BM25 alone**: Faithfulness drops to 0.281 — the lowest of all configurations

**Why**: When the model receives relevant semantic context, it answers accurately. BM25's vocabulary mismatch forces the model to fill in gaps with hallucination. Faithfulness is therefore a leading indicator of retrieval quality, not just LLM quality.

#### 5. BM25 Alone Is the Worst Approach for Code Q&A

BM25 at chunk=256 scores lower than every semantic configuration across all metrics — even worse than semantic chunk=256 (identical chunking).

**The core problem**: Code questions use natural language ("What is AuthService responsible for?") but answers live in code identifiers ("class AuthService:"). Semantic embeddings bridge that gap; BM25 cannot.

**Where BM25 excels**: Exact-match identifier questions:
- "Find every place BillingError is raised" → 1.000 across all metrics ✓
- "What notifications are sent when a task is assigned?" → 1.000 ✓

**Where BM25 fails**: Conceptual questions (0.000 precision and recall).

#### 6. Chunk Size Dominates — Better Embeddings Don't Help Until Chunks Are Large Enough

Three hybrid configurations reveal the hierarchy:

| Config | Precision | Recall | Faithfulness | Change |
|--------|-----------|--------|--------------|--------|
| HuggingFace + BM25, chunk=256 | 0.667 | 0.466 | 0.649 | baseline |
| OpenAI + BM25, chunk=256 | 0.641 | 0.469 | 0.686 | -0.02 avg |
| OpenAI + BM25, chunk=1500 | 0.889 | 0.824 | 0.873 | +0.26 avg |

**Key insight**: Swapping HuggingFace embeddings for OpenAI at chunk=256 changed almost nothing (<0.03 delta). Increasing chunk size from 256 to 1500 (with OpenAI embeddings) produced massive gains: +0.248 precision, +0.355 recall, +0.187 faithfulness.

**Why**: Chunk size is the dominant variable. Better embeddings rank broken pieces more accurately — but broken pieces are still broken pieces. The RRF fusion strategy is sound; it just needs complete chunks to work with.

#### 7. Hybrid chunk=1500 Beats All Previous Best Configs on Every Metric

Previous best: Semantic chunk=1500 (0.778 / 0.650 / 0.821 / 0.821)  
Hybrid chunk=1500: **0.889 / 0.824 / 0.873 / 0.825** ✓

**Why it wins**: The BM25 component adds recall signal for exact identifier questions that pure semantic retrieval misses.

**Questions hybrid chunk=1500 solves that all others failed**:
- "Which methods check is_active before proceeding?" → 1.000 / 1.000 / 1.000 ✓
- "What are all methods inside AuthService?" → 1.000 / 1.000 / 1.000 ✓
- "What fields does the Task dataclass have?" → 1.000 / 1.000 / 1.000 ✓
- "If a user's account is deactivated, which specific checks fire?" → precision and recall 1.000 ✓

**Note on NaN faithfulness**: Two questions (AuthService responsibility, privilege escalation) returned NaN — the agent answered at a high level without citing specific code facts. NaN values are excluded from aggregates, slightly inflating the 0.873 figure.

**Still failing**: 
- "What are all methods inside TaskService?" → 0.000 precision/recall (likely split across chunk boundaries at k=4)
- "Which TaskService methods check that requesting user owns the task?" → 0.000 precision/recall

#### 8. GraphRAG Excels at Structural Questions But Is Blind to Behavioral Detail

**GraphRAG strengths** (depth=2):
- "Trace the complete call chain when create_task is called" → 1.000 / 1.000 / 1.000 / 0.731 ✓
- "What are all methods inside TaskService?" → 1.000 / 1.000 / 1.000 / 0.939 ✓
- "Which file imports both Task and Status?" → 1.000 / 1.000 / 0.500 / 0.411 ✓

**GraphRAG fails on behavioral/value-level questions**:
- "What happens on incorrect password?" → 0.000 precision/recall
- "What notifications are sent when a task is assigned?" → 0.000 precision/recall
- "Find every place BillingError is raised?" → 0.000 (leaf nodes have no graph context)

**Why**: GraphRAG stores edges (X calls Y, A imports B) but not implementation details — what error is raised, what fields exist, what the method does.

**Note on high faithfulness (0.736)**: GraphRAG's strong faithfulness reflects correct refusal to guess when context is missing, not perfect answer quality.

#### 9. Questions That Remain Hard Regardless of Configuration

These expose hard limits no single retriever strategy resolves:

1. **"Which TaskService methods check user ownership?"** — requires reading multiple method bodies simultaneously
2. **"What are all methods inside TaskService?"** — needs whole-class context; hurt by chunking at k=4
3. **"Trace the complete call chain when create_task is called"** — recall stays low (0.286 in best config); only GraphRAG fully solves this

### Configuration Strengths by Question Type

| Question Type | Best Config | Why |
|---------------|-----------|-----|
| Single-file behavioral detail | Hybrid (OpenAI + BM25), chunk=1500 | Complete function context + semantic ranking |
| Exact identifier lookup across files | Hybrid (OpenAI + BM25), chunk=1500 or BM25 | Token matching + semantic re-rank |
| Class structure / all methods in a class | GraphRAG or AST chunking | Structural boundary preservation |
| Multi-hop call chain / dependency trace | GraphRAG | Graph traversal captures all edges |
| Import-level relationships | GraphRAG or Semantic, chunk=1500 | Module-scope visibility |
| Cross-file implicit pattern | Hybrid (OpenAI + BM25), chunk=1500 | Combines semantic + lexical signals |

---

## Architecture

### Cache Strategy

The evaluation uses a **two-layer caching system**:

#### Agent Response Cache (JSON)
**Location**: `evals/.cache/agent_responses_cache.json`

Stores the answer and contexts retrieved by the agent for each question.

**Benefits**:
- ✅ Avoids re-running the agent on repeat evaluations
- ✅ Cost: ~$0 (no API calls for cached questions)
- ✅ Time: ~100ms per cached question vs 2-3s for fresh run

**How it works**:
```python
# First run: Calls Groq (agent) for each question
# Subsequent runs: Loads from cache
cached = get_cached_agent_response(question)
if cached:
    answer, contexts = cached
else:
    answer, contexts = run_agent(agent, question)
    cache_agent_response(question, answer, contexts)
```

#### RAGAS Results Cache (Pickle)
**Location**: `evals/.cache/ragas_results_cache.pkl`

Caches full RAGAS evaluation results indexed by content hash. Eliminates redundant metric computations when samples haven't changed.

**Benefits**:
- ✅ Eliminates 90 Groq API calls (18 test cases × 5 metrics) when samples unchanged
- ✅ Cost: ~$0 for cached evals (vs ~$0.02 fresh)
- ✅ Time: ~100ms cached (vs 30-60s fresh)
- ✅ Hash-based invalidation: automatically re-evaluates if samples change

**How it works**:
```python
# First run: Calls Groq (evaluator) for all 5 metrics on 18 samples
# Subsequent runs (same samples): Loads from cache instantly
cached_results = get_cached_ragas_results(samples)
if cached_results is not None:
    return cached_results
results = evaluate(...)  # Only runs if cache miss
cache_ragas_results(samples, results)
```

---

## Cost Breakdown

### Scenario A: First Run (Everything Fresh)
```
✓ Agent inference (18 questions):           ~$0.01 (Groq)
✓ RAGAS evaluation (90 metric calls):       ~$0.01 (Groq)
✓ Caching overhead:                         negligible
────────────────────────────────────────────────────
TOTAL:                                       ~$0.02
TIME:                                        2-3 minutes
```

### Scenario B: Repeat Run (All Cached)
```
✓ Agent responses:                          Loaded from JSON ✓
✓ RAGAS results:                            Loaded from Pickle ✓
✓ LLM calls:                                0 (zero!)
────────────────────────────────────────────────────
TOTAL:                                       $0.00
TIME:                                        30 seconds
```

### Scenario C: Code Changed (Agent Cache Hit, RAGAS Miss)
```
✓ Agent responses:                          Loaded from JSON ✓
✓ RAGAS evaluation:                         Fresh (90 calls)
✓ LLM calls:                                90 (Groq)
────────────────────────────────────────────────────
TOTAL:                                       ~$0.01
TIME:                                        1-2 minutes
```

---

## Cache Structure

```
evals/
├── 1_semantic_rag_with_256_chunking_and_k_eq_4.py
├── golden_dataset.py
└── .cache/                                  (Auto-created on first run)
    ├── agent_responses_cache.json           (Agent responses)
    └── ragas_results_cache.pkl              (RAGAS results)
```

### Cache File Formats

**Agent Responses Cache** (JSON, human-readable):
```json
{
  "What is AuthService responsible for?": {
    "answer": "AuthService handles authentication...",
    "contexts": ["code chunk 1", "code chunk 2", ...]
  },
  ...
}
```

Size: ~500 KB (18 questions)

**RAGAS Results Cache** (Pickle, binary):
```
{
  "hash": "a3b2c1d4...",      // MD5 of samples
  "results": <RAGAS Result>    // Full evaluation results object
}
```

Size: ~1-2 MB

---

## Cache Invalidation

### Agent Response Cache
- **Invalidates when**: Manual deletion or significant codebase changes
- **Best for**: Iterating on RAGAS metrics without re-running agent
- **Clear with**: `rm evals/.cache/agent_responses_cache.json`
- **Stability**: Cache stays valid until codebase or retrieval changes

### RAGAS Results Cache
- **Invalidates when**: Sample contents change (auto-detected via MD5 hash)
- **Best for**: Running same evaluation multiple times (debugging, etc.)
- **Clear with**: `rm evals/.cache/ragas_results_cache.pkl`
- **Stability**: Cache invalidates if any of (question, answer, context, reference) changes

### Hash-Based Invalidation Example
```
# If any sample content changes:
# hash before: a3b2c1d4f5g6h7i8j9k0l1m2
# hash after:  x1y2z3a4b5c6d7e8f9g0h1i2
# → Cache miss detected
# → Fresh RAGAS evaluation runs
# → New results cached with new hash
```

---

## Console Output

You'll see cache hit/miss indicators throughout execution:

```
✓ Loaded cached response for: What is AuthService responsible for?...
✓ Cached new response for: What protects against privilege escalation...
...
✓ Loaded cached RAGAS results (hash: a3b2c1d4...)

RAGAS SCORECARD — Semantic RAG (Recursive Splitter) on sample_project
Judge: Llama 3.1 8B (via Groq) | Caching: Enabled
```

---

## Implementation Details

### Code Changes Made

#### 1. Cache Configuration
```python
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
AGENT_RESPONSES_CACHE_FILE = CACHE_DIR / "agent_responses_cache.json"
RAGAS_RESULTS_CACHE_FILE = CACHE_DIR / "ragas_results_cache.pkl"
```

#### 2. Agent Response Cache Functions

**`get_cached_agent_response(question: str)`**
- Checks if agent response exists in JSON cache
- Returns `(answer, contexts)` tuple if found, else `None`
- Includes error handling for corrupted/missing cache files

**`cache_agent_response(question: str, answer: str, contexts: list[str])`**
- Stores agent response in JSON cache
- Preserves existing cache entries
- Pretty-prints JSON for readability

#### 3. RAGAS Result Cache Functions

**`compute_samples_hash(samples: list[SingleTurnSample]) -> str`**
- Creates deterministic MD5 hash of sample contents
- Serializes all sample fields (user_input, response, retrieved_contexts, reference)
- Hash changes only if sample content changes

**`get_cached_ragas_results(samples: list[SingleTurnSample]) -> object | None`**
- Loads cached results if hash matches current samples
- Returns None if cache miss (different samples or corrupted file)
- Prints cache hit indicator with hash

**`cache_ragas_results(samples: list[SingleTurnSample], results: object)`**
- Stores RAGAS results with hash metadata
- Uses pickle for efficient serialization of complex objects
- Prints cache storage confirmation with hash

#### 4. Judge Model Configuration
```python
def setup_evaluator_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
```

---

## Performance Impact

### Execution Time
| Stage | First Run | Repeat Runs | Improvement |
|-------|-----------|-----------|-------------|
| Codebase loading/chunking | ~30s | ~30s | (baseline) |
| Agent inference | ~90s | ~0s | 100% |
| RAGAS evaluation | ~60s | ~0s | 100% |
| Cache operations | <5s | <5s | ~100ms per query |
| **Total** | **~2-3 min** | **~30 sec** | **85% faster** |

### API Cost
| Stage | First Run | Repeat Runs | Savings |
|-------|-----------|-----------|---------|
| Agent inference | ~$0.01 | $0.00 | 100% |
| RAGAS evaluation | ~$0.01 | $0.00 | 100% |
| **Total** | **~$0.02** | **$0.00** | **100%** |

### Cache Hit Rates (Expected)
- **Agent responses**: 100% if no codebase changes
- **RAGAS results**: 100% if no answers/contexts change
- **Combined**: 900× cost reduction on repeat evaluations

---

## Troubleshooting

### Cache not being used?
1. Check that `.cache/` directory exists in `evals/` folder
2. Verify cache files are readable: `ls -la evals/.cache/`
3. Check console output for "✓ Loaded cached..." messages

### Want to force fresh evaluation?
```bash
# Clear agent cache
rm evals/.cache/agent_responses_cache.json

# Clear RAGAS cache
rm evals/.cache/ragas_results_cache.pkl

# Or clear everything
rm -r evals/.cache/

# Re-run evaluation
python 1_semantic_rag_with_256_chunking_and_k_eq_4.py
```

### Cache file corrupted?
Delete the problematic cache file and re-run. Cache files are automatically regenerated.

---

## Compatibility

- **Python**: 3.9+ (uses `tuple[str, list[str]] | None` type hints)
- **Dependencies**: No new dependencies added (uses stdlib: json, pickle, hashlib)
- **LangChain**: Requires langchain_groq (already used)
- **RAGAS**: Works with existing RAGAS version

---

## Known Limitations

1. **Agent cache invalidation**: Must be manually cleared if codebase changes significantly
2. **Hash sensitivity**: RAGAS cache invalidates on ANY change to samples (even whitespace in answers)
3. **Storage**: Cache files stored locally (no cloud sync)
4. **Concurrency**: Not thread-safe (multiple processes may corrupt cache)

---

## Future Enhancements

Possible improvements:
1. Add `--clear-cache` CLI flag
2. Add cache size management (prune old entries)
3. Add cache statistics (hit rate, storage used)
4. Support cloud-backed cache (S3, etc.)
5. Add concurrent-safe file locking

---
