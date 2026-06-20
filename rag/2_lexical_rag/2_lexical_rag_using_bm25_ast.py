# Lexical RAG system using BM25 with AST-based chunking
# Combines keyword-based BM25 search with semantically meaningful AST chunks
# BM25 ranks documents based on term frequency and inverse document frequency
# AST chunking ensures each chunk is a complete function or class
# See helpers/ for reusable utility functions

import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation and tool use
from langchain_groq import ChatGroq
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware
from rank_bm25 import BM25Okapi

# Import reusable helpers from root helpers folder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_ast, BM25Retriever


def build_bm25_retriever(chunks: list) -> BM25Retriever:
    """
    Build a BM25 retriever from code chunks.
    Tokenizes all chunks for BM25 indexing.

    BM25 with AST chunks provides:
    - Fast keyword-based retrieval
    - Semantically coherent chunks (complete functions/classes)
    - Better context preservation compared to fixed-size chunks
    """
    # Tokenize all chunks into lowercase words
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    # Create BM25 index from tokenized chunks
    bm25 = BM25Okapi(tokenized)
    # Return a LangChain-compatible BM25 retriever
    return BM25Retriever(docs=chunks, bm25=bm25)


def build_agent(retriever):
    """
    Create an AI agent that uses BM25 retriever for codebase search.
    Agent can answer questions by searching for relevant code chunks.

    The agent combines:
    - BM25 lexical search (fast, keyword-based)
    - AST chunks (semantically coherent, complete functions/classes)
    - Groq LLM (efficient, low-latency reasoning)
    """
    # Create a retriever tool from the BM25 retriever
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic using keyword matching.",
    )
    # Initialize the language model (Groq with Llama 3.1 8B)
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
        # ModelCallLimitMiddleware: Limits LLM calls to 3, then exits gracefully
        # ToolCallLimitMiddleware: Limits codebase search calls to 3, preventing repeated queries
        middleware=[
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=3, exit_behavior="end")
        ]
    )


if __name__ == "__main__":
    # ============ SETUP: Parse command-line arguments ============
    parser = argparse.ArgumentParser(
        description="BM25 + AST-based Lexical RAG for codebase search"
    )
    # Accept a --repo argument, defaulting to the sample_project directory at root
    default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
    parser.add_argument("--repo", default=default_repo, help="Path to codebase to analyze")
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # ============ STEP 1: Load and chunk the codebase using AST ============
    print(f"Loading codebase from: {repo_path}")
    # Load all Python files from the repository
    docs = load_python_codebase(repo_path)
    print(f"Loaded {len(docs)} Python files")

    # Parse code using AST to extract classes and functions (semantically meaningful chunks)
    # Each chunk is a complete class or module-level function
    chunks = chunk_code_ast(docs)
    print(f"Created {len(chunks)} AST-based chunks")
    print(f"Retrieval strategy: BM25 (keyword-based)")
    print(f"Chunking strategy: AST (semantically coherent)")

    # ============ STEP 2: Build BM25 retriever and agent ============
    print("\nBuilding BM25 retriever...")
    # Create a BM25 retriever from the AST chunks
    retriever = build_bm25_retriever(chunks)
    print("BM25 retriever built successfully")

    # Build an AI agent with access to the BM25 search tool
    print("Building agent...")
    agent = build_agent(retriever)
    print("Agent built successfully\n")

    # ============ STEP 3: Interactive Q&A loop ============
    print("=" * 60)
    print("BM25 + AST Lexical RAG System Ready")
    print("=" * 60)
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
