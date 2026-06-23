"""
Semantic RAG Evaluation with OpenAI LLM and RAGAS Metrics

This module implements a Retrieval-Augmented Generation (RAG) system for code analysis with:
- Recursive character-based code chunking (1500-char chunks)
- OpenAI embeddings for semantic search
- OpenAI LLM (gpt-4o-mini) as the reasoning engine
- RAGAS evaluation framework with 5 metrics
- Agent middleware for call/tool usage limits

Workflow: Load code → Chunk → Build vector store → Create agent → Evaluate on test cases
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
# RAGAS expects langchain_community.chat_models.vertexai, so we shim it to langchain_google_vertexai
VERTEXAI_SHIM_PATH = "langchain_community.chat_models.vertexai"
if VERTEXAI_SHIM_PATH not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType(VERTEXAI_SHIM_PATH)
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules[VERTEXAI_SHIM_PATH] = _shim

# LangChain core imports for RAG pipeline
from langchain_core.documents import Document
from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware

# RAGAS evaluation metrics and framework
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import llm_factory

# Import helper functions from shared helpers module for code loading and chunking
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers import load_python_codebase, chunk_code_recursive

# Import test cases for evaluation
from golden_dataset import TEST_CASES

# ============ CONFIGURATION CONSTANTS ============
# Chunk sizing: smaller chunks (1500) for fine-grained retrieval; larger chunks for context preservation
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 32  # Overlap helps maintain context at chunk boundaries

# Retrieval: k=4 means retrieve top 4 most relevant code chunks for each query
RETRIEVAL_K = 4

# Agent middleware: prevent runaway loops and excessive API calls
MAX_MODEL_CALLS = 5  # Max LLM invocations per query
MAX_TOOL_CALLS = 2   # Max retriever tool calls per query

# ============ CACHING CONFIGURATION ============
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
AGENT_RESPONSES_CACHE_FILE = CACHE_DIR / "agent_responses_cache_sm_1500_chunking_k_4.json"
RAGAS_RESULTS_CACHE_FILE = CACHE_DIR / "ragas_results_cache_sm_1500_chunking_k_4.pkl"


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


def build_vector_store_with_openai(chunks: list[Document]) -> Chroma:
    """
    Build vector store using OpenAI embeddings (text-embedding-3-small).

    OpenAI embeddings provide high-quality semantic representations for code.
    This is better than HuggingFace for code understanding due to training on diverse data.

    Args:
        chunks: List of Document objects to embed and index

    Returns:
        Chroma vector store ready for semantic search
    """
    # Use OpenAI's text-embedding-3-small model for cost-efficient, high-quality embeddings
    embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    # Create in-memory Chroma vector store (no persistence to disk)
    return Chroma.from_documents(chunks, embedding=embeddings)


def build_agent(vector_store: Chroma):
    """
    Construct agentic RAG system with OpenAI LLM and semantic search tool.

    The agent uses:
    - OpenAI's gpt-4o-mini LLM for reasoning
    - Retriever tool for semantic search over codebase
    - Middleware to prevent runaway loops

    Args:
        vector_store: Chroma vector store for semantic search

    Returns:
        Agent instance capable of answering queries about the codebase
    """
    # Create retriever tool that searches for k=4 most relevant code chunks
    # This tool is available to the agent during reasoning
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )

    # Initialize OpenAI LLM with gpt-4o-mini model
    # temperature=0 ensures deterministic, reproducible responses
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create agent with system prompt, tools, and middleware
    return create_agent(
        llm,
        tools=[retriever_tool],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering. "
            "Reference specific file and function names. "
            "If not found say 'I could not find that in the codebase'."
        ),
        # Middleware prevents infinite loops and excessive API usage
        middleware=[
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=MAX_TOOL_CALLS, exit_behavior="end")
        ]
    )


def run_agent(agent, question: str) -> tuple[str, list[str]]:
    """
    Execute agent on a question and extract answer with retrieved contexts.

    The agent:
    1. Searches the codebase using the retriever tool
    2. Reasons about the results
    3. Returns an answer

    Args:
        agent: The agent instance
        question: User query about the codebase

    Returns:
        Tuple of (answer_text, list_of_retrieved_contexts)
        Contexts are the code chunks returned by semantic search
    """
    # Invoke agent with the question and collect full message trace
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # Extract the final answer (last message in trace)
    answer = result["messages"][-1].content

    # Extract all tool responses (retrieved code chunks) from the message trace
    # ToolMessages contain the code chunks returned by the retriever
    contexts = []
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage" and isinstance(msg.content, str):
            contexts.append(msg.content)

    return answer, contexts


def build_evaluation_samples(agent, test_cases: list[dict]) -> list[SingleTurnSample]:
    """
    Generate RAGAS evaluation samples by running agent on test cases.

    Implements Local Embedding Cache for agent responses.
    Caches (answer, contexts) per question to avoid re-running agent on repeat evals.

    Each sample contains:
    - user_input: The question
    - response: The agent's answer
    - retrieved_contexts: Code chunks found by semantic search
    - reference: Expected/reference answer for comparison

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
            # Show first 200 chars of each context chunk
            print(f"  [{i}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        print(f"Answer: {answer}")

    return samples


def setup_evaluator_llm():
    """
    Configure RAGAS evaluator LLM using gpt-4o-mini (same as agent).

    Uses same OpenAI API endpoint as the agent for consistency.

    RAGAS evaluates on 5 dimensions:
    - Context precision: Are retrieved chunks relevant to the question?
    - Context recall: Did we retrieve all necessary context?
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer relevancy: Does the answer address the question?
    - Factual correctness: Is the answer factually accurate?

    Returns:
        LLM instance configured for RAGAS evaluation using gpt-4o-mini
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def run_evaluation(samples: list[SingleTurnSample]) -> object:
    """
    Execute RAGAS evaluation metrics on samples.

    Implements: Result Caching for RAGAS evaluation.
    Caches evaluation results by content hash of samples to avoid re-evaluation.

    RAGAS (Retrieval-Augmented Generation Assessment) provides automated evaluation
    using gpt-4o-mini as a judge. It scores the RAG system on 5 dimensions.

    Metrics computed:
    - llm_context_precision_with_reference: Precision of retrieved chunks vs reference
    - context_recall: Coverage of reference answer by retrieved chunks
    - faithfulness: Answer grounded in retrieved context (no hallucinations)
    - answer_relevancy: Answer addresses the user's question
    - factual_correctness: Answer factually accurate (using external knowledge)

    Args:
        samples: List of SingleTurnSample objects to evaluate

    Returns:
        RAGAS evaluation results with metric scores per sample and aggregate
    """
    # Check if RAGAS results are cached first
    cached_results = get_cached_ragas_results(samples)
    if cached_results is not None:
        return cached_results

    print("\nRunning RAGAS evaluation (gpt-4o-mini as judge)...")

    # Setup evaluator LLM for scoring (gpt-4o-mini via OpenAI)
    evaluator_llm = setup_evaluator_llm()
    # Setup embeddings for answer_relevancy metric (semantic similarity scoring)
    lc_embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
    # Create evaluation dataset from samples
    eval_dataset = EvaluationDataset(samples=samples)

    # Run RAGAS evaluation on all 5 metrics
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),  # Compare retrieved vs reference context
            LLMContextRecall(),                   # Coverage of reference by retrieved
            Faithfulness(),                       # Grounding in retrieved context
            AnswerRelevancy(embeddings=lc_embeddings),  # Relevance to question
            FactualCorrectness(),                 # Factual accuracy of answer
        ],
        llm=evaluator_llm,
    )

    # Cache the results for future runs
    cache_ragas_results(samples, results)
    return results


def get_metric_icon(value: float) -> str:
    """
    Return emoji icon based on metric score for visual feedback.

    Scoring thresholds:
    - ✅ Green: value >= 0.7 (Good performance)
    - ⚠️ Yellow: 0.5 <= value < 0.7 (Acceptable, room for improvement)
    - ❌ Red: value < 0.5 (Poor performance)

    Args:
        value: Metric score between 0 and 1

    Returns:
        Emoji icon representing performance level
    """
    if value >= 0.7:
        return "✅"
    return "⚠️ " if value >= 0.5 else "❌"


def print_per_question_results(df, metric_cols: list[str]):
    """
    Print per-question metric breakdown for detailed analysis.

    Shows how well the RAG system performed on each individual test case.
    Useful for identifying which queries are problematic.

    Args:
        df: Results DataFrame from RAGAS evaluation
        metric_cols: List of metric column names to display
    """
    print("\nPer-question breakdown:")
    for _, row in df.iterrows():
        # Show first 65 chars of question for context
        print(f"\n  Q: {row['user_input'][:65]}")
        for col in metric_cols:
            if col in row:
                val = row[col]
                icon = get_metric_icon(val)
                # Format: icon + metric_name + score (3 decimal places)
                print(f"    {icon} {col:<42}: {val:.3f}")


def print_aggregate_results(df, metric_cols: list[str]):
    """
    Print aggregate metric averages across all test cases.

    Shows the overall performance of the RAG system.
    These aggregates help assess whether the system meets quality thresholds.

    Args:
        df: Results DataFrame from RAGAS evaluation
        metric_cols: List of metric column names to average
    """
    print("\nAggregate averages:")
    for col in metric_cols:
        if col in df.columns:
            # Calculate mean score across all test cases
            avg = df[col].mean()
            icon = get_metric_icon(avg)
            # Format: icon + metric_name + average score
            print(f"  {icon} {col:<42}: {avg:.3f}")


def print_scorecard(results):
    """
    Display RAGAS evaluation results as a formatted scorecard.

    Combines per-question and aggregate results into a readable report.
    This is the final output showing RAG system performance.

    Args:
        results: RAGAS evaluation results object
    """
    # Convert results to pandas DataFrame for easy filtering and display
    df = results.to_pandas()
    # Define which RAGAS metrics to display
    metric_cols = [
        "llm_context_precision_with_reference",  # Precision of retrieved context
        "context_recall",                         # Recall of retrieved context
        "faithfulness",                           # Grounding in retrieved context
        "answer_relevancy",                       # Relevance to question
        "factual_correctness",                    # Factual accuracy
    ]

    # Print report header
    print("\nRAGAS SCORECARD — Semantic RAG (Recursive Splitter) on sample_project")
    print("Judge: gpt-4o-mini (via OpenAI) | Caching: Enabled")
    # Print detailed per-question results
    print_per_question_results(df, metric_cols)
    # Print aggregate statistics
    print_aggregate_results(df, metric_cols)
    print()


def main():
    """
    Main workflow orchestrating the semantic RAG evaluation pipeline.

    Pipeline stages:
    1. Load Python codebase from repository
    2. Chunk code using recursive text splitter (1500 chars)
    3. Build vector store with OpenAI embeddings
    4. Create agentic RAG system with OpenAI LLM
    5. Run agent on test cases
    6. Evaluate with RAGAS metrics (5 dimensions)
    7. Display formatted scorecard

    Configuration can be overridden via command-line arguments (--repo).
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Semantic RAG using recursive text splitting and OpenAI LLM"
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).parent.parent / "sample_project"),
        help="Path to repository to analyze (default: sample_project)"
    )
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # ============ STAGE 1: LOAD & PROCESS CODEBASE ============
    print(f"Loading codebase from {repo_path}...")
    # Load all Python files from repository using helper function
    docs = load_python_codebase(repo_path)
    # Chunk code into smaller pieces (1500 chars each) using recursive splitter
    chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    # ============ STAGE 2: BUILD RAG SYSTEM ============
    print("\nBuilding vector store and agent...")
    # Create vector store with OpenAI embeddings for semantic search
    vector_store = build_vector_store_with_openai(chunks)
    # Create agent with OpenAI LLM and retriever tool
    agent = build_agent(vector_store)

    # ============ STAGE 3: GENERATE EVALUATION SAMPLES ============
    print("\nRunning agent on test cases...")
    # Run agent on each test case and collect samples for evaluation
    samples = build_evaluation_samples(agent, TEST_CASES)

    # ============ STAGE 4: RUN RAGAS EVALUATION ============
    # Evaluate samples using RAGAS framework with gpt-4o-mini as judge
    results = run_evaluation(samples)

    # ============ STAGE 5: DISPLAY RESULTS ============
    # Print formatted scorecard with per-question and aggregate metrics
    print_scorecard(results)


if __name__ == "__main__":
    main()
