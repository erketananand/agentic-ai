# Graph RAG (Knowledge Graph Retrieval-Augmented Generation) system for code analysis
# Extracts code relationships (imports, calls, dependencies) and builds a knowledge graph
# Uses LLM to extract structured relationships and answer queries using graph context
# Graph neighbourhood search returns related entities and relationships for RAG context
# See helpers/ for reusable utility functions

import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

# Graph, LangChain, and type-hint imports
import networkx as nx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Import reusable helpers from root helpers folder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers import load_python_codebase, save_graph, load_graph, match_nodes

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default repository path for analysis
DEFAULT_CODEBASE = Path(__file__).parent.parent.parent / "sample_project"

# Graph cache path for persistence
GRAPH_CACHE_PATH = Path(__file__).parent / "llm_graph_cache.graphml"

# File extensions to include in codebase scanning (Python focus, but extensible)
CODE_EXTENSIONS = {".py"}

# Directories to skip during codebase traversal (build artifacts, cache, VCS)
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache"}


# ============================================================================
# 1. PYDANTIC MODELS FOR STRUCTURED EXTRACTION
# ============================================================================

class CodeRelationship(BaseModel):
    """Represents a directed relationship between two code entities.

    Stores structured code relationships extracted by LLM analysis:
    - subject: source entity (class, function, module, file)
    - predicate: relationship type (e.g., IMPORTS, CALLS, DEPENDS_ON)
    - obj: target entity (class, function, module, file)
    """
    subject: str = Field(description="Class, function, module, or file path")
    predicate: str = Field(
        description="Relationship type (e.g., DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, INHERITS_FROM)"
    )
    obj: str = Field(description="Target class, function, module, or file path")


class GraphDocument(BaseModel):
    """Container for all code relationships extracted from a source file.

    Used as response_format for LLM-structured extraction, enabling
    consistent parsing of relationship triples across the entire codebase.
    """
    relationships: list[CodeRelationship] = Field(description="All code relationships extracted from the source file")


class Entities(BaseModel):
    """Container for code entities extracted from a user query.

    Used to identify classes, functions, modules, and file paths
    that the user is asking about, enabling targeted graph retrieval.
    """
    names: list[str] = Field(
        description="Code entities in the query: class names, function names, modules, file paths"
    )


# ============================================================================
# 2. LOAD CODEBASE
# ============================================================================

def load_codebase_with_extensions(root: Path) -> list[tuple[str, str]]:
    """Load all code files from a repository with multi-language support.

    Extends the helpers.load_python_codebase() to support multiple file types
    (Python, TypeScript, JavaScript, Java, Go, Rust, Markdown) and enables
    filtering of build artifacts and cache directories.

    Args:
        root: Path to the repository root

    Returns:
        List of (relative_path, file_content) tuples for all matching files
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path does not exist: {root}")

    files = []
    # Recursively traverse all files in the repository
    for path in sorted(root.rglob("*")):
        # Skip if not a regular file
        if not path.is_file():
            continue
        # Skip if file extension not in CODE_EXTENSIONS
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        # Skip if path contains any excluded directories
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # Add (relative_path, content) tuple to results
        files.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))

    return files


# ============================================================================
# 3. EXTRACT RELATIONSHIPS & BUILD KNOWLEDGE GRAPH
# ============================================================================

def extract_relationships(relationship_extractor, files: list[tuple[str, str]]) -> list[CodeRelationship]:
    """Extract code relationships from all source files using LLM analysis.

    Invokes the relationship_extractor agent on each file to identify
    structured (subject, predicate, object) relationships. Aggregates
    all relationships into a single list for graph construction.

    Args:
        relationship_extractor: LangChain agent configured for relationship extraction
        files: List of (rel_path, content) tuples from load_codebase_with_extensions()

    Returns:
        List of CodeRelationship objects representing all extracted relationships
    """
    relationships = []
    # Process each file through the LLM relationship extractor
    for rel_path, content in files:
        result = relationship_extractor.invoke(
            {"messages": [{"role": "user", "content": f"File: {rel_path}\n\n{content}"}]}
        )
        # Extend aggregated relationships with results from this file
        relationships.extend(result["structured_response"].relationships)

    return relationships


def build_graph(relationships: list[CodeRelationship]) -> nx.DiGraph:
    """Build a directed knowledge graph from extracted code relationships.

    Converts a list of structured relationships into a NetworkX directed graph.
    Each relationship becomes an edge with the predicate stored as relation metadata.

    Args:
        relationships: List of CodeRelationship objects

    Returns:
        NetworkX DiGraph with nodes as entities and edges labeled with predicates
    """
    graph = nx.DiGraph()
    # Add each relationship as a directed edge (subject -> object)
    for r in relationships:
        graph.add_edge(r.subject.strip(), r.obj.strip(), relation=r.predicate)

    return graph


# ============================================================================
# 4. GRAPH RETRIEVAL
# ============================================================================


def graph_retrieve(graph: nx.DiGraph, entity_extractor, query: str, depth: int = 2) -> str:
    """Retrieve relevant code relationships from the knowledge graph using a query.

    Workflow:
    1. Use entity_extractor to identify entities mentioned in the user query
    2. Find matching nodes in the graph (fuzzy match for naming variations)
    3. Expand to ego graphs (neighbourhood at specified depth)
    4. Extract edges from the neighbourhood (relationships with context)
    5. Return formatted relationships as RAG context for the QA agent

    Args:
        graph: NetworkX DiGraph knowledge graph
        entity_extractor: LangChain agent configured for entity extraction
        query: User question containing entity references
        depth: Graph neighbourhood radius (default: 2 hops)

    Returns:
        Formatted string of relevant relationships for RAG context
    """
    # Extract entities from user query using LLM
    result = entity_extractor.invoke({"messages": [{"role": "user", "content": query}]})
    entities = result["structured_response"].names

    # Retrieve relationships from graph neighbourhood
    relationships = []
    for entity in entities:
        # Find nodes matching this entity name
        for node in match_nodes(graph, entity):
            # Get ego graph (neighbourhood) around this node
            neighbourhood = nx.ego_graph(graph, node, radius=depth, undirected=True)
            # Extract and format all edges in the neighbourhood
            for source, target, data in neighbourhood.edges(data=True):
                relationships.append(f"{source} -[{data['relation']}]-> {target}")

    # Return formatted context or empty message
    if not relationships:
        return "No relevant graph data found."
    return "Knowledge Graph context:\n" + "\n".join(sorted(set(relationships)))


# ============================================================================
# 5. MAIN: GRAPH RAG PIPELINE ORCHESTRATION
# ============================================================================

if __name__ == "__main__":
    # ============ SETUP: Parse command-line arguments ============
    parser = argparse.ArgumentParser(
        description="Graph RAG - answer questions about a codebase using a knowledge graph of code relationships"
    )
    parser.add_argument(
        "--repo", type=Path, default=DEFAULT_CODEBASE,
        help="Path to codebase to analyze"
    )
    parser.add_argument(
        "--depth", type=int, default=2,
        help="Graph neighbourhood radius for relationship retrieval (default: 2)"
    )
    args = parser.parse_args()

    # ============ STEP 1: Load codebase (support multi-language) ============
    print(f"Loading codebase from: {args.repo}")
    files = load_codebase_with_extensions(args.repo.resolve())
    if not files:
        raise SystemExit(f"No source files found under {args.repo}")
    print(f"Loaded {len(files)} files")

    # ============ STEP 2: Initialize LLM and agents ============
    # Use GPT LLM for efficient relationship and entity extraction
    gpt_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Use GROQ LLM for Q&A
    groq_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Agent 1: Extract code relationships from source files
    # Converts unstructured code into structured (subject, predicate, object) triples
    relationship_extractor = create_agent(
        model=gpt_llm,
        tools=[],
        response_format=GraphDocument,
        system_prompt=(
            "Extract code relationships from the source file as structured (subject, predicate, object) facts.\n"
            "Format your response to be clear and human-readable.\n\n"
            "Use these predicates for relationships:\n"
            "  DEFINES: When module/file defines a class or function\n"
            "  IMPORTS: When module imports another module or component\n"
            "  CALLS: When function/method calls another function/method\n"
            "  USES: When code uses a class, service, or utility\n"
            "  DEPENDS_ON: When module depends on another module\n"
            "  INHERITS_FROM: When class inherits from a base class\n"
            "  IMPLEMENTS: When class implements an interface\n"
            "  SENDS_TO: When code sends data/messages to external services\n"
            "  CONFIGURES: When code configures or sets up components\n\n"
            "Guidelines:\n"
            "  - Be specific with names: use full paths, class names, function names\n"
            "  - Only include relationships you can directly identify from the code\n"
            "  - Be consistent: use the same name format throughout\n"
            "  - Skip obvious language constructs; focus on architectural relationships"
        ),
    )

    # Agent 2: Extract entities from user queries
    # Identifies classes, functions, modules, file paths mentioned in questions
    entity_extractor = create_agent(
        model=gpt_llm,
        tools=[],
        response_format=Entities,
        system_prompt="Extract code-related entities from the user message: class names, function names, module names, and file paths.",
    )

    # Agent 3: Answer questions using graph context
    # Reasons about code relationships retrieved from the knowledge graph
    qa_agent = create_agent(
        model=groq_llm,
        tools=[],
        system_prompt=(
            "You are a senior codebase assistant helping developers understand code architecture.\n\n"
            "Guidelines for your responses:\n"
            "  - Answer ONLY using the provided knowledge graph context\n"
            "  - Use clear, concise language that non-experts can understand\n"
            "  - Always reference specific file names, class names, and function names\n"
            "  - Explain relationships in plain English before showing the graph notation\n"
            "  - If the context doesn't answer the question, say 'I could not find that in the codebase'\n"
            "  - Organize complex answers with bullet points or numbered lists\n"
            "  - If multiple relationships are relevant, explain how they connect"
        ),
    )

    # ============ STEP 3: Extract relationships and build knowledge graph ============
    # Try to load existing graph from disk
    graph = load_graph(GRAPH_CACHE_PATH)

    if graph is None:
        print("\nExtracting code relationships...")
        relationships = extract_relationships(relationship_extractor, files)
        print(f"Extracted {len(relationships)} relationships")

        # Build NetworkX directed graph from relationships
        graph = build_graph(relationships)
        print(f"Knowledge graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Save graph to disk for reuse
        save_graph(graph, GRAPH_CACHE_PATH)
    else:
        print(f"Using cached graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Display all extracted relationships for transparency
    print("\nKNOWLEDGE GRAPH - All extracted relationships:")
    for source, target, data in sorted(graph.edges(data=True)):
        print(f"  {source:35s} -[{data['relation']:20s}]-> {target}")

    # ============ STEP 4: Interactive Q&A loop ============
    print("\n" + "=" * 70)
    print("Graph RAG System Ready")
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

        # Retrieve relevant knowledge graph context for the question
        context = graph_retrieve(graph, entity_extractor, question, depth=args.depth)
        # Stream QA agent response using graph context
        for step in qa_agent.stream(
            {"messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]},
            stream_mode="values",
        ):
            # Extract and print text responses (skip tool calls)
            last_msg = step["messages"][-1]
            if not getattr(last_msg, "tool_calls", None):
                print(f"Agent: {last_msg.content}\n")
