# Hybrid RAG system combining semantic and lexical retrieval
# Uses Ensemble Retriever with Reciprocal Rank Fusion (RRF) to merge results
# Semantic retriever: Vector similarity using HuggingFace embeddings
# Lexical retriever: BM25 keyword matching
# Weights are customizable (default 0.5/0.5 for equal blending)
# See helpers/ for reusable utility functions

import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation, retrieval, and ensemble
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware

# Import reusable helpers from root helpers folder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_recursive, build_vector_store, BM25Retriever


# ============================================================================
# 1. CHUNKING CONFIGURATION
# ============================================================================

# Chunk size for both semantic and lexical retrievers
CHUNK_SIZE = 500   # Play around with chunk sizes: 250, 500, 1000, 1500


# ============================================================================
# 2. SEMANTIC RETRIEVER: Vector-based similarity search
# ============================================================================

def build_vector_retriever(chunks: list):
    """
    Build a vector-based retriever using embeddings.
    Uses HuggingFace embeddings for semantic similarity search.

    Args:
        chunks: List of Document objects to embed

    Returns:
        LangChain retriever for vector similarity search
    """
    # Build vector store (in-memory) using helper function
    vector_store = build_vector_store(chunks)
    # Return retriever configured to return top-4 most similar chunks
    return vector_store.as_retriever(search_kwargs={"k": 4})


# ============================================================================
# 3. LEXICAL RETRIEVER: BM25 keyword-based search (from helpers)
# ============================================================================

def build_bm25_retriever_hybrid(chunks: list) -> BM25Retriever:
    """
    Build a BM25 retriever for keyword-based search.
    Uses the BM25Retriever class from helpers module.

    Args:
        chunks: List of Document objects to index

    Returns:
        BM25Retriever instance configured with k=4
    """
    # Tokenize all chunks into lowercase words
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    # Create BM25 index from tokenized chunks
    bm25 = BM25Okapi(tokenized)
    # Return a LangChain-compatible BM25 retriever
    return BM25Retriever(docs=chunks, bm25=bm25, k=4)


# ============================================================================
# 4. HYBRID RETRIEVER: Merge semantic and lexical results using RRF
# ============================================================================

def build_hybrid_retriever(chunks: list, semantic_weight: float = 0.5, lexical_weight: float = 0.5):
    """
    Build a hybrid retriever combining semantic and lexical search.
    Uses EnsembleRetriever with Reciprocal Rank Fusion (RRF) to blend results.

    RRF combines rankings from multiple retrievers:
    - Semantic: Finds conceptually similar code
    - Lexical: Finds exact keyword matches

    Args:
        chunks: List of Document objects
        semantic_weight: Weight for semantic retriever (0.0-1.0)
        lexical_weight: Weight for lexical retriever (0.0-1.0)
                       Note: weights should sum to 1.0

    Returns:
        EnsembleRetriever merging both retrievers with RRF

    Example:
        # Equal blending (default)
        retriever = build_hybrid_retriever(chunks)

        # Favor semantic search
        retriever = build_hybrid_retriever(chunks, 0.7, 0.3)

        # Favor lexical search (better for exact identifiers)
        retriever = build_hybrid_retriever(chunks, 0.3, 0.7)
    """
    # Validate weights sum to 1.0
    if not abs(semantic_weight + lexical_weight - 1.0) < 0.01:
        raise ValueError(f"Weights must sum to 1.0, got {semantic_weight} + {lexical_weight} = {semantic_weight + lexical_weight}")

    # Build both retrievers
    vector_retriever = build_vector_retriever(chunks)
    bm25_retriever = build_bm25_retriever_hybrid(chunks)

    # Merge using Ensemble with Reciprocal Rank Fusion
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[semantic_weight, lexical_weight],
    )


# ============================================================================
# 5. AGENT: AI assistant with hybrid search capability
# ============================================================================

def build_agent(retriever):
    """
    Create an AI agent with access to the hybrid retriever.
    Agent can ask follow-up questions and reason about results.

    Args:
        retriever: The hybrid retriever combining semantic and lexical search

    Returns:
        LangChain agent with retriever tool
    """
    # Create a retriever tool for the agent to use
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic using hybrid semantic+lexical search.",
    )
    # Initialize the language model (Groq with Llama 3.1 8B for efficiency)
    # temperature=0 ensures deterministic, focused responses
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    # Create and return an agent with the LLM, retriever tool, and safety middleware
    return create_agent(
        llm, tools=[retriever_tool],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering. "
            "Reference specific file and function names. "
            "If not found say 'I could not find that in the codebase'."
        ),
        # Middleware to prevent infinite loops and excessive API calls
        middleware=[
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=3, exit_behavior="end")
        ]
    )


# ============================================================================
# 6. MAIN: Entry point - orchestrate hybrid RAG pipeline
# ============================================================================

if __name__ == "__main__":
    # ============ SETUP: Parse command-line arguments ============
    parser = argparse.ArgumentParser(
        description="Hybrid RAG combining semantic (embeddings) and lexical (BM25) retrieval"
    )
    # Repository path argument
    default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
    parser.add_argument("--repo", default=default_repo, help="Path to codebase to analyze")
    # Weights for blending retrievers
    parser.add_argument(
        "--weights", nargs=2, type=float, default=[0.5, 0.5],
        metavar=("SEMANTIC", "LEXICAL"),
        help="Retriever blend weights (must sum to 1.0). Default: 0.5 0.5 (equal blending)"
    )
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # Validate weights
    if not abs(sum(args.weights) - 1.0) < 0.01:
        print(f"Error: weights must sum to 1.0, got {args.weights[0]} + {args.weights[1]} = {sum(args.weights)}")
        exit(1)

    # ============ STEP 1: Load and chunk the codebase ============
    print(f"Loading codebase from: {repo_path}")
    # Load all Python files from the repository
    docs = load_python_codebase(repo_path)
    print(f"Loaded {len(docs)} Python files")

    # Split documents using recursive character-based splitting
    chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    # ============ STEP 2: Build hybrid retriever and agent ============
    print(f"\nRetrieval strategy: Hybrid (Semantic {args.weights[0]} + Lexical {args.weights[1]})")
    print("Building hybrid retriever...")

    # Create hybrid retriever with weighted blending
    retriever = build_hybrid_retriever(chunks, semantic_weight=args.weights[0], lexical_weight=args.weights[1])
    print("Hybrid retriever built successfully")

    # Build AI agent with hybrid search capability
    print("Building agent...")
    agent = build_agent(retriever)
    print("Agent built successfully\n")

    # ============ STEP 3: Interactive Q&A loop ============
    print("=" * 70)
    print("Hybrid RAG System Ready (Semantic + Lexical)")
    print("=" * 70)
    print("Ask your question about the codebase")
    print("Type 'exit' or 'quit' to end session\n")

    while True:
        # Get user question
        question = input("You: ").strip()
        # Exit if empty input or user types 'exit'/'quit'
        if not question or question.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        # Stream the agent's response to the question
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            # Get the last message from the agent
            last_msg = step["messages"][-1]
            # Only print if it's a text response (not a tool call)
            if not getattr(last_msg, "tool_calls", None):
                print(f"Agent: {last_msg.content}\n")
