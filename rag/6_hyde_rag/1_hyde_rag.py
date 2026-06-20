# HyDE (Hypothetical Document Embeddings) RAG system
# Generates hypothetical answer documents from queries, then retrieves based on those embeddings
# This approach improves retrieval quality for conceptual/intent-based questions
# See HyDE paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (arXiv:2212.10496)
# Implements HyDE manually to show each step clearly

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation, retrieval, and embeddings
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.embeddings import init_embeddings
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

# Import reusable helpers from root helpers folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_recursive


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Embedding model for semantic similarity
EMBEDDING_MODEL = "text-embedding-3-small"
# LLM model for hypothesis generation and QA
LLM_MODEL = "llama-3.1-8b-instant"
# Code chunk size
CHUNK_SIZE = 1500
# Top-K documents to retrieve
TOP_K = 4




# ============================================================================
# 3. IN-MEMORY VECTOR STORE: Create embeddings index
# ============================================================================

def build_inmemory_vector_store(chunks: list) -> InMemoryVectorStore:
	"""
	Build an in-memory vector store for semantic similarity search.
	Uses OpenAI embeddings (text-embedding-3-small model).

	Args:
		chunks: List of Document objects to embed

	Returns:
		InMemoryVectorStore for similarity search
	"""
	embeddings = init_embeddings(f"openai:{EMBEDDING_MODEL}")
	return InMemoryVectorStore.from_documents(chunks, embeddings)


# ============================================================================
# 2. HYDE GENERATOR: Hypothetical document generation
# ============================================================================

def build_hyde_generator():
	"""
	Create an agent that generates hypothetical answer documents.

	HyDE core idea:
	- Do NOT embed the raw question directly
	- First generate a plausible answer passage (hypothetical document)
	- Then embed that hypothetical document and retrieve nearest real docs
	- This improves retrieval for conceptual queries

	Returns:
		LangChain agent for hypothesis generation
	"""
	return create_agent(
		model=ChatGroq(model=LLM_MODEL, temperature=0),
		tools=[],
		system_prompt=(
			"You generate a single hypothetical code snippet or document for retrieval.\n"
			"Write a compact, plausible-looking passage (4-7 sentences or a short code snippet) "
			"that could answer the question about a Python codebase.\n"
			"Do not add bullet points, no disclaimers, and no markdown."
		),
	)


# ============================================================================
# 3. QA AGENT: HyDE-powered retrieval and question answering
# ============================================================================

def build_qa_agent(vector_store: InMemoryVectorStore):
	"""
	Create the main QA agent with HyDE retrieval capability.
	Internally uses the HyDE generator to create hypothetical documents
	before retrieving from the vector store.

	Args:
		vector_store: InMemoryVectorStore for similarity search

	Returns:
		Tuple of (qa_agent, hyde_agent) for the interactive loop
	"""
	# Initialize the HyDE hypothesis generator
	hyde_agent = build_hyde_generator()

	@tool
	def search_codebase(query: str) -> str:
		"""Retrieve code chunks using HyDE:
		1. Generate a hypothetical answer document from the query
		2. Embed the hypothetical document
		3. Retrieve similar real code chunks from the vector store
		This improves retrieval quality compared to embedding the raw query.
		"""
		# Step 1: Generate hypothetical document
		hypo_response = hyde_agent.invoke(
			{"messages": [{"role": "user", "content": query}]}
		)
		hypothetical_doc = str(hypo_response["messages"][-1].content).strip()
		print(f"\n[HyDE] Hypothetical document:\n{hypothetical_doc}\n")

		# Step 2: Retrieve real documents using hypothetical embedding
		docs = vector_store.similarity_search(hypothetical_doc, k=TOP_K)

		# Format retrieved content
		return "\n\n".join(
			f"# {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
			for doc in docs
		)

	# Create the main QA agent with HyDE retrieval tool
	qa_agent = create_agent(
		model=ChatGroq(model=LLM_MODEL, temperature=0),
		tools=[search_codebase],
		system_prompt=(
			"You are a senior engineer. Always use search_codebase before answering. "
			"Reference specific file and function names. "
			"If not found say 'I could not find that in the codebase'."
		),
		middleware=[
			ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
			ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=2, exit_behavior="end")
		]
	)
	return qa_agent, hyde_agent


# ============================================================================
# 4. MAIN: Entry point - orchestrate HyDE RAG pipeline
# ============================================================================

if __name__ == "__main__":
	# ============ SETUP: Parse command-line arguments ============
	parser = argparse.ArgumentParser(
		description="HyDE RAG system for intelligent codebase retrieval"
	)
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

	# ============ STEP 2: Build in-memory vector store ============
	print("\nBuilding in-memory vector store with OpenAI embeddings...")
	vector_store = build_inmemory_vector_store(chunks)
	print("Vector store built successfully")

	# ============ STEP 3: Build HyDE QA agent ============
	print("\nBuilding HyDE generator and QA agent...")
	qa_agent, hyde_agent = build_qa_agent(vector_store)
	print("HyDE RAG system built successfully\n")

	# ============ STEP 4: Interactive Q&A loop ============
	print("=" * 70)
	print("HyDE RAG System Ready (Hypothetical Document Embeddings)")
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
		for step in qa_agent.stream(
			{"messages": [{"role": "user", "content": question}]},
			stream_mode="values",
		):
			# Get the last message from the agent
			last_msg = step["messages"][-1]
			# Only print if it's a text response (not a tool call)
			if not getattr(last_msg, "tool_calls", None):
				print(f"Agent: {last_msg.content}\n")
