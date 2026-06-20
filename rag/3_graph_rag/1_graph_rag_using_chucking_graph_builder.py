# Graph RAG using deterministic chunking-based graph builder
# Builds knowledge graph from AST-extracted code chunks instead of LLM extraction
# Zero LLM calls during indexing; uses heuristics to derive edges (DEFINES, IMPORTS, CALLS, INHERITS_FROM)
# Graph neighbourhood retrieval provides context for Q&A via Groq LLM

import argparse
from pathlib import Path
from dotenv import load_dotenv

# Graph and LangChain imports
import networkx as nx
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.documents import Document
from pydantic import ConfigDict

# Import reusable helpers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, chunk_code_ast, build_or_load_graph, match_nodes

load_dotenv()

DEFAULT_CODEBASE = Path(__file__).parent.parent.parent / "sample_project"
GRAPH_CACHE_PATH = Path(__file__).parent / "chunking_graph_cache.graphml"




# ============================================================================
# CUSTOM GRAPH RETRIEVER
# ============================================================================

class GraphRetriever(BaseRetriever):
    """Retriever that searches a knowledge graph for relevant chunks.

    Maps query keywords to graph nodes, expands via ego graph, and returns
    associated code chunks for RAG context.
    """
    graph: nx.DiGraph
    chunks_by_name: dict  # entity_name -> list[Document]
    k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Extract keywords from query, find graph nodes, expand ego graph, return chunks."""
        # Simple entity extraction: split query into tokens, match against node names
        tokens = set(query.lower().split())
        matched_nodes = set()

        # Find nodes matching query tokens
        for token in tokens:
            token = token.strip(".,;:!?")
            if len(token) < 2:
                continue
            candidates = match_nodes(self.graph, token)
            matched_nodes.update(candidates)

        # Expand via ego graph
        all_nodes = set()
        for node in matched_nodes:
            ego = nx.ego_graph(self.graph, node, radius=2, undirected=True)
            all_nodes.update(ego.nodes())

        # Collect chunks associated with matched nodes
        relevant_chunks = []
        seen = set()
        for node in all_nodes:
            chunks = self.chunks_by_name.get(node, [])
            for chunk in chunks:
                chunk_id = chunk.metadata.get("source", "") + "_" + chunk.metadata.get("name", "")
                if chunk_id not in seen:
                    relevant_chunks.append(chunk)
                    seen.add(chunk_id)

        return relevant_chunks[: self.k]


def build_graph_retriever(chunks: list[Document], graph: nx.DiGraph) -> GraphRetriever:
    """Build a GraphRetriever from chunks and graph."""
    # Map entity names to their chunks
    chunks_by_name = {}
    for chunk in chunks:
        name = chunk.metadata.get("name")
        source = chunk.metadata.get("source")
        if name:
            chunks_by_name.setdefault(name, []).append(chunk)
        if source:
            chunks_by_name.setdefault(source, []).append(chunk)

    return GraphRetriever(graph=graph, chunks_by_name=chunks_by_name, k=4)


# ============================================================================
# AGENT SETUP
# ============================================================================

def build_agent(retriever: BaseRetriever):
    """Build agent with graph retriever as a search tool."""
    tool = create_retriever_tool(retriever, name="search_codebase",
        description="Search the codebase for relevant code entities and their relationships in the knowledge graph."
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    system_prompt = """You are a senior engineer answering questions about a codebase.
Always use the search_codebase tool to find relevant code.
Reference specific file names, function names, and relationships.
If information is not found in the codebase, say "I could not find that in the codebase"."""

    return create_agent(
        llm,
        tools=[tool],
        system_prompt=system_prompt,
        middleware=[
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=3, exit_behavior="end"),
        ],
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph RAG using chunking-based graph builder (no LLM extraction)"
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_CODEBASE, help="Path to codebase")
    parser.add_argument("--depth", type=int, default=2, help="Graph ego depth (default: 2)")
    parser.add_argument("--no-cache", action="store_true", help="Skip graph cache and rebuild")
    args = parser.parse_args()

    # Load or build graph
    print(f"Loading codebase from: {args.repo}")
    docs = load_python_codebase(args.repo.resolve())
    if not docs:
        raise SystemExit(f"No source files found under {args.repo}")
    print(f"Loaded {len(docs)} files")

    print("Chunking with AST...")
    chunks = chunk_code_ast(docs)
    print(f"Created {len(chunks)} chunks")

    # Load or build graph with caching
    graph = build_or_load_graph(
        chunks,
        GRAPH_CACHE_PATH,
        use_cache=not args.no_cache,
        graph_type="chunking"
    )

    # Build retriever and agent
    print("Building retriever and agent...")
    retriever = build_graph_retriever(chunks, graph)
    agent = build_agent(retriever)

    # Interactive Q&A loop
    print("\n" + "=" * 70)
    print("Graph RAG Ready. Type 'exit' or 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            last_msg = step["messages"][-1]
            if not getattr(last_msg, "tool_calls", None):
                print(f"Agent: {last_msg.content}\n")
