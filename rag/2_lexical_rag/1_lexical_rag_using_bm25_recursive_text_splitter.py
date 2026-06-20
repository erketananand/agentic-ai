# Lexical RAG system using BM25 (Best Matching 25)
# Uses keyword/term-based search (no embeddings) for fast retrieval
# BM25 ranks documents based on term frequency and inverse document frequency
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
from helpers import load_python_codebase, chunk_code_recursive, BM25Retriever

# Chunk size for BM25 (keyword-based) - play around with 256, 500, 1000, 1500
CHUNK_SIZE = 1200


def build_bm25_retriever(chunks: list) -> BM25Retriever:
    """
    Build a BM25 retriever from code chunks.
    Tokenizes all chunks for BM25 indexing.
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
    """
    # Create a retriever tool from the BM25 retriever
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
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
        middleware=[
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=3, exit_behavior="end")
        ]
    )


if __name__ == "__main__":
    # ============ SETUP: Parse command-line arguments ============
    parser = argparse.ArgumentParser()
    # Accept a --repo argument, defaulting to the sample_project directory at root
    default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
    parser.add_argument("--repo", default=default_repo)
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # ============ STEP 1: Load and chunk the codebase ============
    # Load all Python files from the repository
    docs = load_python_codebase(repo_path)
    # Split documents into smaller chunks for BM25 indexing
    chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    # ============ STEP 2: Build BM25 retriever and agent ============
    # Create a BM25 retriever from the code chunks
    retriever = build_bm25_retriever(chunks)
    # Build an AI agent with access to the BM25 search tool
    agent = build_agent(retriever)

    # ============ STEP 3: Interactive Q&A loop ============
    print("Ready. Ask your question. Type 'exit' to quit")
    while True:
        # Get user question
        question = input("\nYou: ").strip()
        # Exit if empty input or user types 'exit'/'quit'
        if not question or question.lower() in ("exit", "quit"):
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
                print(f"Agent: {last_msg.content}")
