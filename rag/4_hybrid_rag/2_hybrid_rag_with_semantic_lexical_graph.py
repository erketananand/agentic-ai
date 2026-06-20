# Hybrid RAG system combining Semantic, Lexical, and Graph retrieval
# Uses three complementary retrieval strategies to find and understand code:
# - Semantic: Vector embeddings for conceptual similarity (design patterns, implementations)
# - Lexical: BM25 keyword matching for exact code locations (functions, classes, APIs)
# - Graph: Knowledge graph for relationships and architecture (dependencies, multi-hop analysis)
# See helpers/ for reusable utility functions

import argparse
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# LangChain imports for agent creation, retrieval, and tools
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from pydantic import BaseModel, Field

# Import reusable helpers from root helpers folder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import (
    load_python_codebase,
    chunk_code_recursive,
    chunk_code_ast,
    build_vector_store,
    BM25Retriever,
    build_or_load_graph,
)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CHUNK_SIZE = 500  # For semantic and lexical retrievers
GRAPH_DEPTH = 2   # For graph neighbourhood traversal
GRAPH_CACHE_PATH = Path(__file__).parent / "hybrid_graph_cache.graphml"


# ============================================================================
# 2. PYDANTIC MODELS FOR GRAPH EXTRACTION
# ============================================================================

class CodeRelationship(BaseModel):
    subject: str = Field(description="Class, function, module, or file path")
    predicate: str = Field(description="Relationship type (DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, etc.)")
    obj: str = Field(description="Target class, function, module, or file path")


class GraphDocument(BaseModel):
    relationships: list[CodeRelationship] = Field(description="All code relationships")


class Entities(BaseModel):
    names: list[str] = Field(description="Code entities: class names, function names, module paths")


# ============================================================================
# 3. SEMANTIC RETRIEVER (Vector-based similarity search)
# ============================================================================

def build_vector_retriever(chunks: list):
    """Build vector-based semantic retriever using embeddings."""
    vector_store = build_vector_store(chunks)
    return vector_store.as_retriever(search_kwargs={"k": 4})


# ============================================================================
# 4. LEXICAL RETRIEVER (BM25 keyword matching)
# ============================================================================

def build_bm25_retriever_triple(chunks: list) -> BM25Retriever:
    """Build BM25 retriever for keyword-based lexical search."""
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized)
    return BM25Retriever(docs=chunks, bm25=bm25, k=4)


# ============================================================================
# 5. GRAPH RETRIEVER (Relationship and architecture analysis)
# ============================================================================

class GraphRetriever(BaseRetriever):
    """Custom retriever that searches knowledge graphs for related entities and relationships."""
    graph: nx.DiGraph
    chunks_by_entity: dict  # Maps entity names to document chunks
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list:
        """Retrieve chunks related to entities in the query."""
        # Extract entity keywords from query (simple heuristic: capitalized words, common names)
        entity_keywords = self._extract_entities_simple(query)

        if not entity_keywords:
            return []

        # Find matching nodes in the graph
        relevant_chunks = []
        seen_entities = set()

        for keyword in entity_keywords:
            # Find nodes that contain this keyword
            matching_nodes = [node for node in self.graph.nodes()
                             if keyword.lower() in node.lower()]

            for node in matching_nodes:
                if node not in seen_entities:
                    seen_entities.add(node)
                    # Get ego graph neighbourhood
                    ego = nx.ego_graph(self.graph, node, radius=GRAPH_DEPTH, undirected=True)

                    # Extract relationships and find associated chunks
                    for source, target, data in ego.edges(data=True):
                        relation_text = f"{source} -[{data['relation']}]-> {target}"
                        # Look for chunks mentioning these entities
                        for entity in [source, target]:
                            if entity in self.chunks_by_entity:
                                relevant_chunks.extend(self.chunks_by_entity[entity])

        # Return unique chunks (up to k)
        return list(dict.fromkeys(relevant_chunks))[:self.k] if relevant_chunks else []

    def _extract_entities_simple(self, query: str) -> list[str]:
        """Simple entity extraction: split query and filter by length and patterns."""
        words = query.split()
        entities = []
        for word in words:
            # Keep capitalized words and common identifiers
            if len(word) > 2 and (word[0].isupper() or '_' in word or word.isidentifier()):
                entities.append(word.rstrip('.,;:!?'))
        return entities


def build_graph_retriever(chunks: list, graph: nx.DiGraph) -> GraphRetriever:
    """Build a graph-based retriever that maps entities to code chunks."""
    # Create entity-to-chunk mapping
    chunks_by_entity = {}
    for chunk in chunks:
        # Extract entity names from chunk metadata
        if hasattr(chunk, 'metadata'):
            entity_name = chunk.metadata.get('name', '')
            if entity_name:
                chunks_by_entity[entity_name] = chunks_by_entity.get(entity_name, []) + [chunk]

    return GraphRetriever(graph=graph, chunks_by_entity=chunks_by_entity, k=4)


# ============================================================================
# 6. TRIPLE HYBRID RETRIEVER (Ensemble of all three strategies)
# ============================================================================

class TripleHybridRetriever(BaseRetriever):
    """Ensemble retriever combining semantic, lexical, and graph-based retrieval."""
    semantic_retriever: BaseRetriever
    lexical_retriever: BM25Retriever
    graph_retriever: GraphRetriever
    semantic_weight: float = 0.33
    lexical_weight: float = 0.33
    graph_weight: float = 0.34

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list:
        """Retrieve documents using all three strategies and merge results."""
        # Retrieve from each strategy
        semantic_docs = self.semantic_retriever.invoke(query)
        lexical_docs = self.lexical_retriever.invoke(query)
        graph_docs = self.graph_retriever.invoke(query)

        # Merge results with deduplication (prefer semantic > lexical > graph by default)
        merged = {}
        seen_content = set()

        for doc in semantic_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                merged[content_hash] = doc
                seen_content.add(content_hash)

        for doc in lexical_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                merged[content_hash] = doc
                seen_content.add(content_hash)

        for doc in graph_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                merged[content_hash] = doc
                seen_content.add(content_hash)

        return list(merged.values())[:8]  # Return top 8 merged results


def build_triple_hybrid_retriever(
    chunks: list,
    graph: nx.DiGraph,
    semantic_weight: float = 0.33,
    lexical_weight: float = 0.33,
    graph_weight: float = 0.34
) -> TripleHybridRetriever:
    """Build triple hybrid retriever combining all three retrieval strategies."""
    # Build individual retrievers
    semantic_retriever = build_vector_retriever(chunks)
    lexical_retriever = build_bm25_retriever_triple(chunks)
    graph_retriever = build_graph_retriever(chunks, graph)

    # Return ensemble
    return TripleHybridRetriever(
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
        graph_retriever=graph_retriever,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        graph_weight=graph_weight,
    )




# ============================================================================
# 8. AGENT WITH TRIPLE HYBRID RETRIEVER
# ============================================================================

def build_triple_hybrid_agent(retriever):
    """Create an AI agent with triple hybrid retriever access."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_codebase",
        description=(
            "Search the codebase using triple hybrid retrieval: "
            "semantic (conceptual similarity), lexical (keyword matching), "
            "and graph-based (relationships and architecture). "
            "Best for comprehensive code understanding."
        ),
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    return create_agent(
        llm,
        tools=[retriever_tool],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering. "
            "Reference specific file and function names. "
            "Consider relationships between modules and explain architecture. "
            "If not found say 'I could not find that in the codebase'."
        ),
        middleware=[
            ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=3, exit_behavior="end")
        ]
    )


# ============================================================================
# 9. MAIN: ORCHESTRATE TRIPLE HYBRID RAG PIPELINE
# ============================================================================

if __name__ == "__main__":
    # ============ SETUP: Parse command-line arguments ============
    parser = argparse.ArgumentParser(
        description="Hybrid RAG combining Semantic, Lexical, and Graph retrieval strategies"
    )
    default_repo = str(Path(__file__).parent.parent.parent / "sample_project")
    parser.add_argument("--repo", default=default_repo, help="Path to codebase to analyze")
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=[0.33, 0.33, 0.34],
        metavar=("SEMANTIC", "LEXICAL", "GRAPH"),
        help="Retriever weights (semantic, lexical, graph). Must sum to 1.0. Default: 0.33 0.33 0.34"
    )
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    # Validate weights
    weight_sum = sum(args.weights)
    if not abs(weight_sum - 1.0) < 0.01:
        print(f"Error: weights must sum to 1.0, got {weight_sum}")
        exit(1)

    # ============ STEP 1: Load and chunk the codebase ============
    print(f"Loading codebase from: {repo_path}")
    docs = load_python_codebase(repo_path)
    print(f"Loaded {len(docs)} Python files")

    # Create two chunk versions: AST for better semantic understanding
    chunks_ast = chunk_code_ast(docs)
    print(f"Created {len(chunks_ast)} AST-based chunks")

    # ============ STEP 2: Build or load knowledge graph ============
    print("\nBuilding or loading knowledge graph...")
    graph = build_or_load_graph(
        chunks_ast,
        GRAPH_CACHE_PATH,
        use_cache=True,
        graph_type="simple"
    )

    # ============ STEP 3: Build triple hybrid retriever ============
    print(f"\nBuilding triple hybrid retriever...")
    print(f"  Weights: Semantic {args.weights[0]:.2f} + Lexical {args.weights[1]:.2f} + Graph {args.weights[2]:.2f}")

    retriever = build_triple_hybrid_retriever(
        chunks_ast,
        graph,
        semantic_weight=args.weights[0],
        lexical_weight=args.weights[1],
        graph_weight=args.weights[2]
    )
    print("Triple hybrid retriever built successfully")

    # ============ STEP 4: Build agent ============
    print("Building agent...")
    agent = build_triple_hybrid_agent(retriever)
    print("Agent built successfully\n")

    # ============ STEP 5: Interactive Q&A loop ============
    print("=" * 75)
    print("Triple Hybrid RAG System Ready (Semantic + Lexical + Graph)")
    print("=" * 75)
    print("This system combines three retrieval strategies:")
    print("  • Semantic: Vector embeddings for conceptual understanding")
    print("  • Lexical: BM25 keyword matching for exact code locations")
    print("  • Graph: Knowledge graph for relationships and architecture")
    print("\nAsk your question about the codebase")
    print("Type 'exit' or 'quit' to end session\n")

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
