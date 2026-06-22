# Hybrid RAG system combining semantic and lexical retrieval
# Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers
# Semantic retriever: Vector similarity using HuggingFace embeddings
# Lexical retriever: BM25 keyword matching
# RRF score = sum(1 / (k + rank_i)) across all retrievers, k=60 is the smoothing constant
# Standard RRF treats all retrievers equally
# See helpers/ for reusable utility functions

import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation and retrieval
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
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
# 4. RRF FUSION: Custom Reciprocal Rank Fusion retriever
# ============================================================================

class RRFRetriever(BaseRetriever):
    """
    Reciprocal Rank Fusion (RRF) retriever combining multiple ranked lists.

    Standard RRF Algorithm (Cormack et al., 2009):
        For each document d appearing in any retriever's ranked list:
            rrf_score(d) = sum over retrievers of: 1 / (k + rank_i(d))

        where:
            rank_i(d) = 1-based position of d in retriever i's results
                        (documents not in a retriever's list get no contribution)
            k          = smoothing constant (default 60) that dampens the
                         advantage of high-ranking documents, ensuring no single
                         retriever can dominate purely by rank placement

    RRF treats all retrievers equally and is proven to be more robust than
    score-based fusion across heterogeneous retriever types.

    Final results are sorted by descending RRF score and the top_k are returned.

    Attributes:
        retrievers: List of BaseRetriever instances
        k_constant: RRF smoothing constant (default 60)
        top_k: Number of final documents to return (default 4)
    """

    retrievers: list   # list of BaseRetriever instances
    k_constant: int = 60
    top_k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Fuse ranked lists from all retrievers via standard RRF and return top_k results.

        Steps:
            1. Collect ranked results from each retriever independently.
            2. For every (retriever, document) pair compute the RRF score contribution.
            3. Accumulate RRF scores keyed by document content (documents appearing
               in multiple retrievers have additive contributions).
            4. Sort descending by accumulated RRF score and return top_k documents.
        """
        # rrf_scores[doc_content] -> (cumulative_rrf_score, Document)
        # keyed by content so documents appearing in multiple retrievers are merged
        rrf_scores: dict[str, tuple[float, Document]] = {}

        for retriever in self.retrievers:
            results = retriever.invoke(query)
            for rank, doc in enumerate(results, start=1):
                # Standard RRF contribution: 1 / (k + rank)
                contribution = 1.0 / (self.k_constant + rank)
                key = doc.page_content
                if key in rrf_scores:
                    prev_score, existing_doc = rrf_scores[key]
                    rrf_scores[key] = (prev_score + contribution, existing_doc)
                else:
                    rrf_scores[key] = (contribution, doc)

        # Sort by accumulated RRF score descending and return top_k
        ranked = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.top_k]]


def build_hybrid_retriever(chunks: list):
    """
    Build a hybrid RRF retriever combining semantic and lexical search.
    Uses standard Reciprocal Rank Fusion (treats both retrievers equally).

    Args:
        chunks: List of Document objects

    Returns:
        RRFRetriever fusing semantic (vector) and lexical (BM25) retrievers

    Example:
        retriever = build_hybrid_retriever(chunks)
    """
    vector_retriever = build_vector_retriever(chunks)
    bm25_retriever = build_bm25_retriever_hybrid(chunks)

    return RRFRetriever(
        retrievers=[vector_retriever, bm25_retriever]
    )


# ============================================================================
# 5. AGENT: AI assistant with hybrid search capability
# ============================================================================

def build_agent(retriever):
    """
    Create an AI agent with access to the RRF hybrid retriever.
    Agent uses Reciprocal Rank Fusion to combine semantic and lexical results.

    Args:
        retriever: The RRFRetriever combining semantic (vector) and lexical (BM25) search

    Returns:
        LangChain agent with retriever tool
    """
    # Create a retriever tool for the agent to use
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description="Search the codebase using RRF (Reciprocal Rank Fusion) over semantic embeddings and BM25 lexical matches.",
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
        description="Hybrid RAG using Reciprocal Rank Fusion (RRF) combining semantic and lexical retrieval"
    )
    # Repository path argument
    default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
    parser.add_argument("--repo", default=default_repo, help="Path to codebase to analyze")
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # ============ STEP 1: Load and chunk the codebase ============
    print(f"Loading codebase from: {repo_path}")
    # Load all Python files from the repository
    docs = load_python_codebase(repo_path)
    print(f"Loaded {len(docs)} Python files")

    # Split documents using recursive character-based splitting
    chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    # ============ STEP 2: Build RRF hybrid retriever and agent ============
    print("\nRetrieval strategy: Reciprocal Rank Fusion (RRF)")
    print("Building RRF retriever (combining vector similarity + BM25 keyword search)...")

    # Create RRF retriever combining semantic and lexical retrievers equally
    retriever = build_hybrid_retriever(chunks)
    print("RRF retriever built successfully")

    # Build AI agent with hybrid search capability
    print("Building agent...")
    agent = build_agent(retriever)
    print("Agent built successfully\n")

    # ============ STEP 3: Interactive Q&A loop ============
    print("=" * 70)
    print("RRF Hybrid RAG System Ready (Reciprocal Rank Fusion: Semantic + Lexical)")
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
