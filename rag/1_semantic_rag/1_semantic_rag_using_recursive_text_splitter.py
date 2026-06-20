# Semantic RAG (Retrieval-Augmented Generation) system for code analysis
# Uses embeddings-based semantic search to understand code semantically
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

# Import reusable helpers from root helpers folder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_recursive, build_vector_store

# Size of each chunk when splitting code (controls granularity for embeddings)
CHUNK_SIZE = 1200  # Play around with chunk sizes: 500, 1000, 1500


def build_agent(vector_store):
    """
    Create an AI agent that can search the codebase and answer questions about it.
    The agent uses an LLM with access to a retriever tool for semantic search.
    """
    # Create a retriever tool that searches for k=4 most relevant code chunks
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 4}),
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )
    # Initialize the language model (Groq with Llama 3.1 8B, temperature=0 for deterministic responses)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    # Create and return an agent with the LLM and retriever tool
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
    # Split documents into smaller chunks for better embeddings (recursive character-based)
    chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE)
    print(f"Loaded {len(docs)} files → {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    # ============ STEP 2: Build the vector store and agent ============
    # Create a vector database from the code chunks (in-memory)
    # For persistence, use: build_vector_store(chunks, persist_directory="./chroma_db")
    vector_store = build_vector_store(chunks)
    # Build an AI agent with access to the codebase search tool
    agent = build_agent(vector_store)

    # ============ STEP 3: Interactive loop - accept user questions ============
    print("Ready. Ask your question. Type 'exit' to quit")
    while True:
        # Get user input
        question = input("\nYou: ").strip()
        # Exit if user types nothing or 'exit'/'quit'
        if not question or question.lower() in ("exit", "quit"):
            break

        # Stream the agent's response to the question
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            # Get the last message from the agent
            last_msg = step["messages"][-1]
            # Only print the response if it's not a tool call (actual text response)
            if not getattr(last_msg, "tool_calls", None):
                print(f"Agent: {last_msg.content}")
