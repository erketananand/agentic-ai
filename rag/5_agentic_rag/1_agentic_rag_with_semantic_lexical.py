# Agentic RAG system with semantic and lexical retrieval routing
# Uses a router agent to intelligently select between two retrieval strategies
# Semantic retriever: Vector similarity using OpenAI embeddings
# Lexical retriever: BM25 keyword matching
# Router agent: LLM-powered decision making on which retriever to use
# See helpers/ for reusable utility functions

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation, retrieval, and tools
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

# Import reusable helpers from root helpers folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_recursive


# ============================================================================
# 1. CHUNKING CONFIGURATION
# ============================================================================

# Chunk size for both semantic and lexical retrievers
CHUNK_SIZE = 1500




# ============================================================================
# 2. SEMANTIC RETRIEVER: Vector-based similarity search
# ============================================================================

def build_vector_store(chunks: list) -> Chroma:
	"""
	Build a vector store for semantic similarity search.
	Uses OpenAI embeddings (text-embedding-3-small model).

	Args:
		chunks: List of Document objects to embed

	Returns:
		Chroma vector store for similarity search
	"""
	embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
	return Chroma.from_documents(chunks, embedding=embeddings)


# ============================================================================
# 3. LEXICAL RETRIEVER: BM25 keyword-based search
# ============================================================================

def build_bm25_index(chunks: list) -> tuple:
	"""
	Build a BM25 index for keyword-based retrieval.
	BM25 (Best Matching 25) ranks documents by keyword relevance.

	Args:
		chunks: List of Document objects to index

	Returns:
		Tuple of (BM25Okapi instance, original chunks list)
	"""
	tokenized = [doc.page_content.lower().split() for doc in chunks]
	return BM25Okapi(tokenized), chunks


# ============================================================================
# 4. ROUTER AGENT: Intelligent retrieval selection
# ============================================================================
# Architecture:
#   retrieval_tool (outer @tool)
#       └─ router_agent (LLM-powered decision making)
#               ├─ semantic_retrieval  (dense vector search via Chroma)
#               └─ lexical_retrieval   (sparse BM25 keyword search)
#
# The router agent analyzes the query and decides which retrieval
# method is more appropriate for answering the question.

def build_retrieval_tool(vector_store: Chroma, bm25: BM25Okapi, bm25_docs: list):
	"""
	Create a retrieval tool that intelligently routes queries to
	semantic or lexical retrievers based on query intent.

	Architecture: Outer tool delegates to a router agent that internally
	calls semantic or lexical retrieval methods as appropriate.

	Args:
		vector_store: Chroma vector store for semantic search
		bm25: BM25Okapi index for lexical search
		bm25_docs: Original documents indexed by BM25

	Returns:
		A tool that can be used by an outer agent
	"""
	# Initialize the router LLM (Groq with Llama 3.1 8B for efficiency)
	# temperature=0 ensures deterministic, focused decisions
	llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

	@tool
	def semantic_retrieval(query: str) -> str:
		"""Retrieve code chunks using dense vector (semantic) similarity.
		Use for conceptual or intent-based queries about how something works."""
		docs = vector_store.similarity_search(query, k=4)
		print(f"  [semantic_retrieval] '{query}' → {len(docs)} chunks")
		return "\n\n".join(
			f"# {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs
		)

	@tool
	def lexical_retrieval(query: str) -> str:
		"""Retrieve code chunks using BM25 keyword matching.
		Use for exact identifiers: function names, class names, error strings."""
		tokens = query.lower().split()
		scores = bm25.get_scores(tokens)
		top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
		docs = [bm25_docs[i] for i in top_indices]
		print(f"  [lexical_retrieval]  '{query}' → {len(docs)} chunks")
		return "\n\n".join(
			f"# {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs
		)

	# Create a router agent that decides which retrieval method to use
	router_agent = create_agent(
		model=llm,
		tools=[semantic_retrieval, lexical_retrieval],
		system_prompt=(
			"You are a retrieval router for a Python codebase. "
			"Given a query, call the right retrieval tool:\n"
			"- semantic_retrieval: conceptual queries ('how does X work?', 'where is Y handled?')\n"
			"- lexical_retrieval: exact names or strings ('find ClassName', 'where is func_name called?')\n"
			"Return all retrieved content without summarising it."
		),
		middleware=[
			ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
		]
	)

	@tool
	def retrieval_tool(query: str) -> str:
		"""Retrieve relevant code from the codebase for any query.
		Internally routes to semantic or lexical retrieval as appropriate."""
		print(f"\n[retrieval_tool] Routing query: '{query}'")
		result = router_agent.invoke(
			{"messages": [{"role": "user", "content": query}]}
		)
		return str(result["messages"][-1].content)

	return retrieval_tool


# ============================================================================
# 5. MAIN AGENT: AI assistant with retrieval capability
# ============================================================================

def build_agent(retrieval_tool):
	"""
	Create the main AI agent with access to the retrieval tool.
	This agent answers questions about the codebase by first retrieving
	relevant code using the intelligent routing retrieval tool.

	Args:
		retrieval_tool: The routing retrieval tool

	Returns:
		LangChain agent ready to answer questions
	"""
	# Initialize the language model (Groq with Llama 3.1 8B for efficiency)
	# temperature=0 ensures deterministic, focused responses
	llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
	# Create and return an agent with the LLM, retrieval tool, and safety middleware
	return create_agent(
		model=llm,
		tools=[retrieval_tool],
		system_prompt=(
			"You are a senior engineer. Always use retrieval_tool before answering. "
			"Reference specific file and function names. "
			"If not found say 'I could not find that in the codebase'."
		),
		middleware=[
			ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
			ToolCallLimitMiddleware(tool_name="retriever_tool", run_limit=2, exit_behavior="end")
		]
	)


# ============================================================================
# 6. MAIN: Entry point - orchestrate agentic RAG pipeline
# ============================================================================

if __name__ == "__main__":
	# ============ SETUP: Parse command-line arguments ============
	parser = argparse.ArgumentParser(
		description="Agentic RAG with semantic and lexical retrieval routing"
	)
	# Repository path argument
	default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
	parser.add_argument("--repo", default=default_repo, help="Path to codebase to analyze")
	args = parser.parse_args()
	repo_path = str(Path(args.repo).resolve())

	# ============ STEP 1: Load and chunk the codebase ============
	print(f"Loading codebase from: {repo_path}")
	# Load all Python files from the repository using helper function
	docs = load_python_codebase(repo_path)
	print(f"Loaded {len(docs)} Python files")

	# Split documents using recursive character-based splitting using helper function
	chunks = chunk_code_recursive(docs, chunk_size=CHUNK_SIZE)
	print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

	# ============ STEP 2: Build vector store and BM25 index ============
	print("\nBuilding semantic and lexical indexes...")
	vector_store = build_vector_store(chunks)
	print("Vector store built (OpenAI embeddings)")

	bm25, bm25_docs = build_bm25_index(chunks)
	print("BM25 index built (keyword search)")

	# ============ STEP 3: Build retrieval tool and agent ============
	print("\nBuilding router agent and retrieval tool...")
	retrieval_tool = build_retrieval_tool(vector_store, bm25, bm25_docs)
	agent = build_agent(retrieval_tool)
	print("Agent built successfully\n")

	# ============ STEP 4: Interactive Q&A loop ============
	print("=" * 70)
	print("Agentic RAG System Ready (Semantic + Lexical Routing)")
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
