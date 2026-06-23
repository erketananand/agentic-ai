"""
Hybrid RAG Evaluation with OpenAI LLM and RAGAS Metrics

This module implements a hybrid Retrieval-Augmented Generation (RAG) system combining:
- Semantic retrieval (dense vector embeddings with OpenAI text-embedding-3-small)
- Lexical retrieval (sparse BM25 keyword matching)
- Reciprocal Rank Fusion (RRF) to blend results from both retrievers
- Recursive character-based code chunking (1500-char chunks)
- OpenAI LLM (gpt-4o-mini) as the reasoning engine
- RAGAS evaluation framework with 5 metrics
- Agent middleware for call/tool usage limits
- Local embedding cache for agent responses
- Result caching for RAGAS evaluation

Workflow: Load code → Chunk → Build hybrid retriever → Create agent → Evaluate on test cases
"""

import sys
import types
import argparse
import json
import pickle
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Inject VertexAI shim for RAGAS compatibility
VERTEXAI_SHIM_PATH = "langchain_community.chat_models.vertexai"
if VERTEXAI_SHIM_PATH not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType(VERTEXAI_SHIM_PATH)
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules[VERTEXAI_SHIM_PATH] = _shim

# LangChain core imports for RAG pipeline
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from rank_bm25 import BM25Okapi

# RAGAS evaluation metrics and framework
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import llm_factory

# Import helper functions for code loading and chunking
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_python_codebase, chunk_code_recursive, BM25Retriever

# Import test cases for evaluation
from golden_dataset import TEST_CASES

# ============ CONFIGURATION CONSTANTS ============
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 32

# Retrieval: k=4 means retrieve top 4 most relevant code chunks for each query
RETRIEVAL_K = 4

# Hybrid retriever weights (semantic vs lexical)
# [0.5, 0.5] means equal importance to both retrievers
HYBRID_WEIGHTS = [0.5, 0.5]

# Agent middleware: prevent runaway loops and excessive API calls
MAX_MODEL_CALLS = 5
MAX_TOOL_CALLS = 2

# ============ CACHING CONFIGURATION ============
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
AGENT_RESPONSES_CACHE_FILE = CACHE_DIR / "agent_responses_cache_hybrid_1500_chunking_k_4.json"
RAGAS_RESULTS_CACHE_FILE = CACHE_DIR / "ragas_results_cache_hybrid_1500_chunking_k_4.pkl"


# ============ CACHE HELPER FUNCTIONS ============

def get_cached_agent_response(question: str) -> tuple[str, list[str]] | None:
    """
    Retrieve cached agent response for a question (Local Embedding Cache).

    Returns:
        Tuple of (answer, contexts) if cached, None otherwise
    """
    if not AGENT_RESPONSES_CACHE_FILE.exists():
        return None
    try:
        cache = json.load(open(AGENT_RESPONSES_CACHE_FILE))
        if question in cache:
            cached = cache[question]
            return cached["answer"], cached["contexts"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def cache_agent_response(question: str, answer: str, contexts: list[str]):
    """Store agent response in local cache for future runs."""
    cache = {}
    if AGENT_RESPONSES_CACHE_FILE.exists():
        try:
            cache = json.load(open(AGENT_RESPONSES_CACHE_FILE))
        except json.JSONDecodeError:
            pass
    cache[question] = {"answer": answer, "contexts": contexts}
    with open(AGENT_RESPONSES_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def compute_samples_hash(samples: list[SingleTurnSample]) -> str:
    """Compute hash of samples for RAGAS result caching."""
    samples_str = json.dumps(
        [
            {
                "user_input": s.user_input,
                "response": s.response,
                "retrieved_contexts": s.retrieved_contexts,
                "reference": s.reference,
            }
            for s in samples
        ],
        sort_keys=True,
    )
    return hashlib.md5(samples_str.encode()).hexdigest()


def get_cached_ragas_results(samples: list[SingleTurnSample]) -> object | None:
    """
    Retrieve cached RAGAS evaluation results (Result Caching).

    Results are cached by content hash of samples to avoid re-evaluation
    when samples haven't changed.
    """
    if not RAGAS_RESULTS_CACHE_FILE.exists():
        return None
    try:
        samples_hash = compute_samples_hash(samples)
        with open(RAGAS_RESULTS_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("hash") == samples_hash:
            print(f"✓ Loaded cached RAGAS results (hash: {samples_hash[:8]}...)")
            return cache["results"]
    except (pickle.UnpicklingError, KeyError, TypeError):
        pass
    return None


def cache_ragas_results(samples: list[SingleTurnSample], results: object):
    """Store RAGAS evaluation results in cache indexed by samples hash."""
    samples_hash = compute_samples_hash(samples)
    with open(RAGAS_RESULTS_CACHE_FILE, "wb") as f:
        pickle.dump({"hash": samples_hash, "results": results}, f)
    print(f"✓ Cached RAGAS results (hash: {samples_hash[:8]}...)")


# ============ RETRIEVER BUILDERS ============

def build_vector_retriever(chunks: list[Document]):
    """
    Build semantic retriever using OpenAI embeddings.

    Uses text-embedding-3-small for cost-efficient, high-quality embeddings.
    """
    embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def build_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    """Build lexical retriever using BM25 keyword matching."""
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)
    return BM25Retriever(docs=chunks, bm25=bm25, k=RETRIEVAL_K)


def build_hybrid_retriever(chunks: list[Document], weights: list = None):
    """
    Build hybrid retriever combining semantic and lexical search via Reciprocal Rank Fusion.

    Args:
        chunks: List of Document objects
        weights: Blend weights [semantic_weight, lexical_weight] (default: [0.5, 0.5])

    Returns:
        EnsembleRetriever combining vector and BM25 retrievers
    """
    vector_retriever = build_vector_retriever(chunks)
    bm25_retriever = build_bm25_retriever(chunks)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights or HYBRID_WEIGHTS,
    )


# ============ AGENT BUILDERS ============

def build_agent(retriever):
    """
    Construct agentic RAG system with OpenAI LLM and hybrid retriever tool.

    The agent uses:
    - OpenAI's gpt-4o-mini LLM for reasoning
    - Hybrid retriever tool combining semantic and lexical search
    - Middleware to prevent runaway loops

    Args:
        retriever: Hybrid retriever combining semantic and lexical search

    Returns:
        Agent instance capable of answering queries about the codebase
    """
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(
        llm,
        tools=[retriever_tool],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering. "
            "Reference specific file and function names. "
            "If not found say 'I could not find that in the codebase'."
        ),
        middleware=[
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=MAX_TOOL_CALLS, exit_behavior="end")
        ]
    )


def run_agent(agent, question: str) -> tuple[str, list[str]]:
    """
    Execute agent on a question and extract answer with retrieved contexts.

    Args:
        agent: The agent instance
        question: User query about the codebase

    Returns:
        Tuple of (answer_text, list_of_retrieved_contexts)
    """
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    contexts = []
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage" and isinstance(msg.content, str):
            contexts.append(msg.content)

    return answer, contexts


# ============ EVALUATION HELPERS ============

def build_evaluation_samples(agent, test_cases: list[dict]) -> list[SingleTurnSample]:
    """
    Generate RAGAS evaluation samples by running agent on test cases.

    Implements Local Embedding Cache for agent responses.
    Caches (answer, contexts) per question to avoid re-running agent on repeat evals.

    Args:
        agent: Agent instance to evaluate
        test_cases: List of dicts with keys: question, reference

    Returns:
        List of SingleTurnSample objects for RAGAS evaluation
    """
    samples = []
    for tc in test_cases:
        question = tc["question"]

        # Check local cache first before running agent
        cached = get_cached_agent_response(question)
        if cached:
            answer, contexts = cached
            print(f"\n✓ Loaded cached response for: {question[:60]}...")
        else:
            # Run agent and get answer + retrieved contexts
            answer, contexts = run_agent(agent, question)
            # Cache this response for future runs
            cache_agent_response(question, answer, contexts)
            print(f"\n✓ Cached new response for: {question[:60]}...")

        # Create RAGAS sample for this query
        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=tc["reference"],
        ))

        # Print detailed output for debugging and visibility
        print(f"Q: {question}")
        print(f"Contexts retrieved ({len(contexts)}):")
        for i, ctx in enumerate(contexts, 1):
            print(f"  [{i}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        print(f"Answer: {answer}")

    return samples


def setup_evaluator_llm():
    """Configure RAGAS evaluator LLM using gpt-4o-mini."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def run_evaluation(samples: list[SingleTurnSample]) -> object:
    """
    Execute RAGAS evaluation metrics on samples.

    Implements: Result Caching for RAGAS evaluation.
    Caches evaluation results by content hash of samples to avoid re-evaluation.

    Args:
        samples: List of SingleTurnSample objects to evaluate

    Returns:
        RAGAS evaluation results with metric scores
    """
    # Check if RAGAS results are cached first
    cached_results = get_cached_ragas_results(samples)
    if cached_results is not None:
        return cached_results

    print("\nRunning RAGAS evaluation (gpt-4o-mini as judge)...")

    # Setup evaluator LLM and embeddings
    evaluator_llm = setup_evaluator_llm()
    lc_embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    eval_dataset = EvaluationDataset(samples=samples)

    # Run RAGAS evaluation on all 5 metrics
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            AnswerRelevancy(embeddings=lc_embeddings),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
    )

    # Cache the results for future runs
    cache_ragas_results(samples, results)
    return results


def get_metric_icon(value: float) -> str:
    """Return emoji icon based on metric score for visual feedback."""
    if value >= 0.7:
        return "✅"
    return "⚠️ " if value >= 0.5 else "❌"


def print_per_question_results(df, metric_cols: list[str]):
    """Print per-question metric breakdown for detailed analysis."""
    print("\nPer-question breakdown:")
    for _, row in df.iterrows():
        print(f"\n  Q: {row['user_input'][:65]}")
        for col in metric_cols:
            if col in row:
                val = row[col]
                icon = get_metric_icon(val)
                print(f"    {icon} {col:<42}: {val:.3f}")


def print_aggregate_results(df, metric_cols: list[str]):
    """Print aggregate metric averages across all test cases."""
    print("\nAggregate averages:")
    for col in metric_cols:
        if col in df.columns:
            avg = df[col].mean()
            icon = get_metric_icon(avg)
            print(f"  {icon} {col:<42}: {avg:.3f}")


def print_scorecard(results):
    """Display RAGAS evaluation results as a formatted scorecard."""
    df = results.to_pandas()
    metric_cols = [
        "llm_context_precision_with_reference",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "factual_correctness",
    ]

    print("\nRAGAS SCORECARD — Hybrid RAG (Semantic + Lexical) on sample_project")
    print("Judge: gpt-4o-mini (via OpenAI) | Caching: Enabled")
    print_per_question_results(df, metric_cols)
    print_aggregate_results(df, metric_cols)
    print()


def main():
    """
    Main workflow orchestrating the hybrid RAG evaluation pipeline.

    Pipeline stages:
    1. Load Python codebase from repository
    2. Chunk code using recursive text splitter (1500 chars)
    3. Build hybrid retriever combining semantic and lexical search
    4. Create agentic RAG system with OpenAI LLM
    5. Run agent on test cases with response caching
    6. Evaluate with RAGAS metrics (5 dimensions)
    7. Display formatted scorecard
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Hybrid RAG combining semantic and lexical retrieval with OpenAI LLM"
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).parent.parent / "sample_project"),
        help="Path to repository to analyze (default: sample_project)"
    )
    parser.add_argument(
        "--weights",
        nargs=2,
        type=float,
        default=HYBRID_WEIGHTS,
        metavar=("SEMANTIC", "LEXICAL"),
        help="Retriever blend weights (must sum to 1.0). Default: 0.5 0.5",
    )
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # ============ STAGE 1: LOAD & PROCESS CODEBASE ============
    print(f"Loading codebase from {repo_path}...")
    docs = load_python_codebase(repo_path)
    chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")
    print(f"Hybrid weights — semantic: {args.weights[0]}, lexical: {args.weights[1]}")

    # ============ STAGE 2: BUILD RAG SYSTEM ============
    print("\nBuilding hybrid retriever and agent...")
    retriever = build_hybrid_retriever(chunks, weights=args.weights)
    agent = build_agent(retriever)

    # ============ STAGE 3: GENERATE EVALUATION SAMPLES ============
    print("\nRunning agent on test cases...")
    samples = build_evaluation_samples(agent, TEST_CASES)

    # ============ STAGE 4: RUN RAGAS EVALUATION ============
    results = run_evaluation(samples)

    # ============ STAGE 5: DISPLAY RESULTS ============
    print_scorecard(results)


if __name__ == "__main__":
    main()
