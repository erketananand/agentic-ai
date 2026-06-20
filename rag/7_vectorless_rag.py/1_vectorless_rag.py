# PageIndex RAG system — vectorless retrieval via agentic tree navigation
# No embeddings. No vector store. Pure LLM reasoning over a hierarchical index.
# Based on: https://github.com/VectifyAI/PageIndex — "similarity ≠ relevance, retrieval requires reasoning"
#
# Tree structure (3 levels):
#   directory (branch) — LLM summary
#   └─ file (branch) — LLM summary
#      └─ chunk (leaf) — first line as summary, source as content
#
# Query time: agent reasons over tree using tools to drill into relevant files
# and fetch only the chunks it needs.

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation, tools, and text splitting
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Import reusable helpers from root helpers folder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_recursive


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# LLM model for summarization and QA
LLM_MODEL = "llama-3.1-8b-instant"
# Code chunk size
CHUNK_SIZE = 1500
# Cache directory for persisted PageIndex trees
CACHE_DIR = Path(__file__).parent / ".pageindex_cache"


# ============================================================================
# 2. PERSISTENCE: Save and load PageIndex trees
# ============================================================================

def _get_index_cache_path(repo_path: str) -> Path:
	"""
	Get the cache file path for a given repository.
	Uses a hash of the repo path to create unique cache files.

	Args:
		repo_path: Path to the repository

	Returns:
		Path to the cache file
	"""
	import hashlib
	repo_hash = hashlib.md5(str(Path(repo_path).resolve()).encode()).hexdigest()[:8]
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	return CACHE_DIR / f"index_{repo_hash}.json"


def save_page_index(index: dict, repo_path: str) -> None:
	"""
	Persist the PageIndex tree to disk as JSON.

	Args:
		index: Root node of the PageIndex tree
		repo_path: Path to the repository (for cache naming)
	"""
	cache_path = _get_index_cache_path(repo_path)
	try:
		with open(cache_path, "w") as f:
			json.dump(index, f, indent=2)
		print(f"✓ Index cached to {cache_path}")
	except Exception as e:
		print(f"⚠ Failed to save index cache: {e}")


def load_page_index(repo_path: str) -> dict | None:
	"""
	Load a previously persisted PageIndex tree from disk.

	Args:
		repo_path: Path to the repository (for cache naming)

	Returns:
		Root node of the PageIndex tree, or None if cache doesn't exist
	"""
	cache_path = _get_index_cache_path(repo_path)
	if not cache_path.exists():
		return None
	try:
		with open(cache_path, "r") as f:
			index = json.load(f)
		print(f"✓ Index loaded from cache: {cache_path}")
		return index
	except Exception as e:
		print(f"⚠ Failed to load index cache: {e}")
		return None


# ============================================================================
# 3. PAGEINDEX TREE BUILDING: Hierarchical summarization
# ============================================================================

def _llm_summary(llm, prompt: str) -> str:
	"""
	Get a single-turn LLM summary of a prompt.

	Args:
		llm: ChatGroq LLM instance
		prompt: Prompt for summarization

	Returns:
		Trimmed summary response
	"""
	return llm.invoke(prompt).content.strip()


def _build_chunk_nodes(file_node_id: str, source: str) -> list[dict]:
	"""
	Split a Python file into chunks and create leaf nodes for the tree.
	Each chunk becomes a leaf — the agent fetches these by node_id at query time.

	Args:
		file_node_id: Unique ID for the parent file node
		source: Python source code

	Returns:
		List of leaf node dictionaries (chunks)
	"""
	# Split the file into chunks using Python-aware separators (class/def boundaries first).
	splitter = RecursiveCharacterTextSplitter.from_language(
		language=Language.PYTHON,
		chunk_size=CHUNK_SIZE,
		chunk_overlap=32
	)
	chunks = splitter.split_text(source)
	return [
		{
			"title": f"chunk_{i + 1}",
			"node_id": f"{file_node_id}::chunk_{i + 1}",
			"summary": chunk.splitlines()[0][:120],  # first line as lightweight preview
			"is_leaf": True,
			"content": chunk,  # actual source — only exposed when agent calls get_chunk_content
			"nodes": [],
		}
		for i, chunk in enumerate(chunks)
	]


def _build_tree(path: Path, base: Path, llm) -> dict | None:
	"""
	Recursively build a hierarchical tree index of the codebase.
	Each directory and file node contains an LLM-generated summary.

	Args:
		path: Current directory or file to process
		base: Repository root (for relative paths)
		llm: ChatGroq LLM instance for summarization

	Returns:
		Tree node dictionary or None if path contains no Python files
	"""
	rel = str(path.relative_to(base)).replace("\\", "/")

	if path.is_file() and path.suffix == ".py":
		source = path.read_text(encoding="utf-8", errors="ignore")
		print(f"  Summarising {rel} ...")

		# LLM reads the file and writes a summary stored in the index.
		# The agent sees this summary (not the source) when browsing get_structure().
		summary = _llm_summary(
			llm,
			f"In 1-2 sentences, summarise this Python file for a search index.\n"
			f"File: {rel}\n\n```python\n{source[:3000]}\n```",
		)

		# File is a branch node — its children are the source chunks.
		chunks = _build_chunk_nodes(rel, source)
		return {
			"title": path.name,
			"node_id": rel,
			"summary": summary,
			"is_leaf": False,
			"nodes": chunks,
		}

	if path.is_dir():
		# Recurse into subdirectories; skip hidden dirs and __pycache__.
		children = [
			node
			for child in sorted(path.iterdir())
			if not child.name.startswith(".") and child.name != "__pycache__"
			if (node := _build_tree(child, base, llm)) is not None
		]
		if not children:
			return None

		# Directory summary is derived from the names of its children (no source needed).
		summary = _llm_summary(
			llm,
			f"In 1 sentence, summarise this Python package for a search index.\n"
			f"Package: {path.name}\nContains: {', '.join(c['title'] for c in children)}",
		)
		return {
			"title": path.name,
			"node_id": rel or "root",
			"summary": summary,
			"is_leaf": False,
			"nodes": children,
		}

	return None


def build_page_index(repo_path: str) -> dict:
	"""
	Build the full hierarchical PageIndex tree.
	One LLM call per file/directory for summarization.

	Args:
		repo_path: Path to the repository root

	Returns:
		Root node of the complete tree
	"""
	base = Path(repo_path)
	llm = ChatGroq(model=LLM_MODEL, temperature=0)
	root = _build_tree(base, base, llm) or {
		"title": base.name,
		"node_id": "root",
		"summary": "Empty project",
		"is_leaf": False,
		"nodes": [],
	}
	root["node_id"] = "root"
	root["title"] = base.name
	return root


def _count_leaves(node: dict) -> int:
	"""
	Count total number of leaf nodes (chunks) in the tree.

	Args:
		node: Tree node to traverse

	Returns:
		Total count of leaf nodes
	"""
	if node["is_leaf"]:
		return 1
	return sum(_count_leaves(c) for c in node.get("nodes", []))


def _find_node(node: dict, node_id: str) -> dict | None:
	"""
	Depth-first search to find a node by its ID.

	Args:
		node: Tree node to start search from
		node_id: ID of node to find

	Returns:
		Node dictionary or None if not found
	"""
	if node["node_id"] == node_id:
		return node
	for child in node.get("nodes", []):
		found = _find_node(child, node_id)
		if found:
			return found
	return None


def _tree_to_str(node: dict, indent: int = 0) -> str:
	"""
	Render the tree showing only node_id + summary (no source code).
	This is what the agent reads via get_structure() to decide what to fetch.

	Args:
		node: Tree node to render
		indent: Indentation level

	Returns:
		String representation of the tree
	"""
	prefix = "  " * indent
	icon = "📄" if node["is_leaf"] else "📁"
	lines = [
		f"{prefix}{icon} [{node['node_id']}] {node['title']}",
		f"{prefix}   {node['summary']}"
	]
	for child in node.get("nodes", []):
		lines.append(_tree_to_str(child, indent + 1))
	return "\n".join(lines)


# ============================================================================
# 4. PAGEINDEX AGENT: Tree-based reasoning retrieval
# ============================================================================

def build_pageindex_agent(index: dict):
	"""
	Create an agent that reasons over the PageIndex tree to retrieve relevant chunks.
	The agent browses the tree structure and fetches only needed chunks.

	Args:
		index: Root node of the PageIndex tree

	Returns:
		LangChain agent for PageIndex-based retrieval
	"""
	# Initialize the language model (Groq with Llama 3.1 8B for efficiency)
	# temperature=0 ensures deterministic, focused retrieval decisions
	llm = ChatGroq(model=LLM_MODEL, temperature=0)

	@tool
	def get_structure() -> str:
		"""Get the full codebase tree with summaries.
		Call this first to find relevant chunks by browsing the hierarchy."""
		return _tree_to_str(index)

	@tool
	def get_chunk_content(node_id: str) -> str:
		"""Get the source code of a specific chunk by node_id (must be a leaf node).
		Fetch only chunks you've identified as relevant from get_structure()."""
		node = _find_node(index, node_id)
		if node is None:
			return f"node_id '{node_id}' not found. Call get_structure() for valid IDs."
		if not node["is_leaf"]:
			children = [c["node_id"] for c in node.get("nodes", [])]
			return f"'{node_id}' is a branch. Children: {children}"
		return node.get("content", "(empty)")

	# Create and return the agent with tree navigation tools and middleware
	return create_agent(
		model=llm,
		tools=[get_structure, get_chunk_content],
		system_prompt=(
			"You are a senior engineer using PageIndex to search a codebase.\n"
			"Always call get_structure() first to find relevant chunk node_ids.\n"
			"Then call get_chunk_content(node_id) only for relevant chunks.\n"
			"Answer from retrieved content only. Reference file and function names.\n"
			"If not found say 'I could not find that in the codebase'."
		),
		middleware=[
			ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
			ToolCallLimitMiddleware(tool_name="get_chunk_content", run_limit=4, exit_behavior="end"),
		],
	)


# ============================================================================
# 5. MAIN: Entry point - orchestrate PageIndex RAG pipeline
# ============================================================================

if __name__ == "__main__":
	# ============ SETUP: Parse command-line arguments ============
	parser = argparse.ArgumentParser(
		description="PageIndex RAG: agentic tree navigation, no vectors"
	)
	default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
	parser.add_argument("--repo", default=default_repo, help="Path to codebase to analyze")
	parser.add_argument("--rebuild", action="store_true", help="Force rebuild index (ignore cache)")
	args = parser.parse_args()
	repo_path = str(Path(args.repo).resolve())

	# ============ STEP 1: Load or build PageIndex tree ============
	print(f"Repository: {repo_path}\n")

	# Try to load from cache if not forcing rebuild
	if not args.rebuild:
		index = load_page_index(repo_path)
		if index:
			total_chunks = _count_leaves(index)
			print(f"Index loaded: {total_chunks} chunks — no vectors, no embeddings\n")
		else:
			print("Cache not found. Building new index...\n")
			index = build_page_index(repo_path)
			total_chunks = _count_leaves(index)
			print(f"\nPageIndex ready: {total_chunks} chunks — no vectors, no embeddings")
			save_page_index(index, repo_path)
			print()
	else:
		print("Rebuilding index (--rebuild flag set)...\n")
		index = build_page_index(repo_path)
		total_chunks = _count_leaves(index)
		print(f"\nPageIndex ready: {total_chunks} chunks — no vectors, no embeddings")
		save_page_index(index, repo_path)
		print()

	# ============ STEP 2: Display tree structure ============
	print("=" * 70)
	print("Codebase Structure (Summaries Only)")
	print("=" * 70)
	print(_tree_to_str(index))

	# ============ STEP 3: Build PageIndex agent ============
	print("\nBuilding PageIndex agent...")
	agent = build_pageindex_agent(index)
	print("Agent built successfully\n")

	# ============ STEP 4: Interactive Q&A loop ============
	print("=" * 70)
	print("PageIndex RAG System Ready (Tree Navigation + LLM Reasoning)")
	print("=" * 70)
	print("Ask your question about the codebase")
	print("Type 'exit' or 'quit' to end session")
	print("Use --rebuild flag to force rebuild index\n")

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
