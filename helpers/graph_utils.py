"""Graph utilities for building and querying knowledge graphs."""

from pathlib import Path
import re
import networkx as nx
from langchain_core.documents import Document


def save_graph(graph: nx.DiGraph, save_path: Path) -> None:
    """Save knowledge graph to disk using GraphML format.

    Persists the graph for later reuse without re-extracting relationships.

    Args:
        graph: NetworkX DiGraph to save
        save_path: Path to save the graph file (.graphml)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(save_path))
    print(f"Graph saved to: {save_path}")


def load_graph(save_path: Path) -> nx.DiGraph | None:
    """Load knowledge graph from disk.

    Args:
        save_path: Path to the saved graph file (.graphml)

    Returns:
        NetworkX DiGraph loaded from disk, or None if file doesn't exist
    """
    if save_path.exists():
        graph = nx.read_graphml(str(save_path))
        print(f"Graph loaded from: {save_path}")
        return graph
    return None


def match_nodes(graph: nx.DiGraph, entity: str) -> list[str]:
    """Find graph nodes that fuzzy-match a given entity name.

    Performs bidirectional substring matching (case-insensitive) to find
    nodes that either contain the entity name or are contained within it.
    Useful for matching entities despite naming variations.

    Args:
        graph: NetworkX DiGraph to search
        entity: Entity name to match (e.g., 'User', 'auth_service', 'models.py')

    Returns:
        List of matching node identifiers from the graph
    """
    needle = entity.strip().lower()
    return [node for node in graph.nodes() if needle in node.lower() or node.lower() in needle]


def build_chunking_graph_from_chunks(chunks: list[Document]) -> nx.DiGraph:
    """Build a deterministic knowledge graph from AST-chunked code.

    Creates nodes for modules and entities, then derives 4 edge types:
    1. DEFINES: module → function/class (from chunk metadata)
    2. IMPORTS: detect 'import X' / 'from X import' patterns
    3. CALLS: detect known entity names appearing in other chunks
    4. INHERITS_FROM: detect base classes in class definitions

    Args:
        chunks: List of Document chunks with metadata['source'], ['name'], ['type']

    Returns:
        NetworkX DiGraph with multi-type edges
    """
    graph = nx.DiGraph()
    entity_names = set()
    chunks_by_source = {}

    # Phase 1: Create module nodes and entity nodes, extract entity names
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        if source:
            graph.add_node(source, type="module")
            chunks_by_source.setdefault(source, []).append(chunk)

        entity_name = chunk.metadata.get("name")
        entity_type = chunk.metadata.get("type")
        if entity_name:
            graph.add_node(entity_name, type=entity_type, source=source)
            entity_names.add(entity_name)
            if source:
                graph.add_edge(source, entity_name, relation="DEFINES")

    # Phase 2: IMPORTS edges - scan for import statements
    import_patterns = [
        r"^import\s+([\w.]+)",
        r"^from\s+([\w.]+)\s+import",
    ]
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        if not source:
            continue
        content = chunk.page_content
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                continue
            for pattern in import_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    module = match.split(".")[0]
                    candidates = match_nodes(graph, module)
                    for candidate in candidates:
                        graph.add_edge(source, candidate, relation="IMPORTS")

    # Phase 3: CALLS edges - detect entity names appearing in other chunks
    for chunk in chunks:
        content = chunk.page_content
        entity_name = chunk.metadata.get("name")
        if not entity_name:
            continue
        for other_entity in entity_names:
            if other_entity == entity_name:
                continue
            if re.search(rf"\b{re.escape(other_entity)}\s*\(", content):
                graph.add_edge(entity_name, other_entity, relation="CALLS")

    # Phase 4: INHERITS_FROM edges - detect class inheritance
    for chunk in chunks:
        entity_type = chunk.metadata.get("type")
        if entity_type != "class":
            continue
        entity_name = chunk.metadata.get("name")
        content = chunk.page_content
        match = re.search(rf"class\s+{re.escape(entity_name)}\s*\(([\w, ]+)\)", content)
        if match:
            bases = match.group(1).split(",")
            for base in bases:
                base = base.strip()
                candidates = match_nodes(graph, base)
                for candidate in candidates:
                    graph.add_edge(entity_name, candidate, relation="INHERITS_FROM")

    return graph


def build_simple_graph_from_chunks(chunks: list[Document]) -> nx.DiGraph:
    """Build a simple knowledge graph from chunk metadata and imports.

    Lightweight alternative to chunking graph: creates nodes from chunk metadata,
    then adds IMPORTS edges by detecting module references in content.

    Args:
        chunks: List of Document chunks with metadata

    Returns:
        NetworkX DiGraph with nodes and IMPORTS edges
    """
    graph = nx.DiGraph()

    for chunk in chunks:
        if hasattr(chunk, "metadata") and "name" in chunk.metadata:
            name = chunk.metadata["name"]
            source = chunk.metadata.get("source", "")
            graph.add_node(name, type=chunk.metadata.get("type", "unknown"), source=source)

    for chunk in chunks:
        if hasattr(chunk, "metadata") and "name" in chunk.metadata:
            source_name = chunk.metadata["name"]
            content = chunk.page_content.lower()
            if "import" in content:
                for other_chunk in chunks:
                    if hasattr(other_chunk, "metadata") and "name" in other_chunk.metadata:
                        target_name = other_chunk.metadata["name"]
                        if target_name != source_name and target_name.lower() in content:
                            graph.add_edge(source_name, target_name, relation="IMPORTS")

    return graph


def build_or_load_graph(
    chunks: list[Document],
    cache_path: Path,
    use_cache: bool = True,
    graph_type: str = "chunking"
) -> nx.DiGraph:
    """Build or load a graph from cache.

    Attempts to load a cached graph first (if use_cache=True and file exists).
    If cache miss or use_cache=False, builds the graph and saves it.

    Args:
        chunks: List of Document chunks for graph building
        cache_path: Path to save/load cached graph
        use_cache: Whether to use cached graph if available (default: True)
        graph_type: "chunking" for deterministic 4-edge graph, "simple" for lightweight graph

    Returns:
        NetworkX DiGraph (either loaded from cache or newly built)
    """
    # Try loading from cache if enabled
    if use_cache:
        graph = load_graph(cache_path)
        if graph is not None:
            return graph

    # Build graph
    print("Building graph from chunks...")
    if graph_type == "chunking":
        graph = build_chunking_graph_from_chunks(chunks)
    else:
        graph = build_simple_graph_from_chunks(chunks)

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Save to cache
    save_graph(graph, cache_path)

    return graph
