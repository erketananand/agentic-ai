"""
Graph RAG Evaluation with OpenAI LLM and RAGAS Metrics

This module implements a Graph-based Retrieval-Augmented Generation (RAG) system for code analysis with:
- Code relationship extraction using OpenAI LLM
- Knowledge graph construction using NetworkX
- Entity-based graph retrieval
- RAGAS evaluation framework with 5 metrics
- Comprehensive caching for graph building, agent responses, and evaluation results

Workflow: Load code → Extract relationships → Build graph → Query graph → Evaluate with RAGAS
"""

import sys
import types
import argparse
import json
import pickle
import hashlib
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community 0.4. Inject a shim before ragas loads.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim


import networkx as nx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from pydantic import BaseModel, Field


from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.llms import llm_factory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CODEBASE = PROJECT_ROOT / "sample_project"

CODE_EXTENSIONS = {".py", ".ts", ".js", ".java", ".go", ".rs", ".md"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache"}

# ============ CACHING CONFIGURATION ============
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
GRAPH_CACHE_FILE = CACHE_DIR / "llm_graph_cache.graphml"
AGENT_RESPONSES_CACHE_FILE = CACHE_DIR / "agent_responses_cache_graph_rag.json"
RAGAS_RESULTS_CACHE_FILE = CACHE_DIR / "ragas_results_cache_graph_rag.pkl"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CodeRelationship(BaseModel):
    subject: str = Field(description="Class, function, module, or file path")
    predicate: str = Field(
        description="Relationship type (e.g., DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, INHERITS_FROM)"
    )
    obj: str = Field(description="Target class, function, module, or file path")


class GraphDocument(BaseModel):
    relationships: list[CodeRelationship] = Field(description="All code relationships extracted from the source file")


class Entities(BaseModel):
    names: list[str] = Field(
        description="Code entities in the query: class names, function names, modules, file paths"
    )


# ============ CACHE HELPER FUNCTIONS ============

def _load_graph_from_graphml(cache_file: Path) -> nx.DiGraph | None:
    """Load NetworkX graph from GraphML cache file."""
    if not cache_file.exists():
        return None
    try:
        return nx.read_graphml(str(cache_file))
    except (nx.NetworkXError, FileNotFoundError):
        return None


def _save_graph_to_graphml(cache_file: Path, graph: nx.DiGraph):
    """Save NetworkX graph to GraphML cache file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(cache_file))


def _load_pickle_cache(cache_file: Path) -> dict | None:
    """Load pickle cache file, return None on error."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, KeyError, TypeError):
        return None


def _save_pickle_cache(cache_file: Path, data: dict):
    """Save dict to pickle cache file."""
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def get_cached_graph(files_hash: str) -> nx.DiGraph | None:
    """
    Retrieve cached graph (Graph Building Cache using GraphML format).

    Returns:
        NetworkX DiGraph if cached, None otherwise
    """
    graph = _load_graph_from_graphml(GRAPH_CACHE_FILE)
    if graph is not None:
        print(f"✓ Loaded cached graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        return graph
    return None


def cache_graph(files_hash: str, graph: nx.DiGraph, relationships: list):
    """Store graph in cache using GraphML format."""
    _save_graph_to_graphml(GRAPH_CACHE_FILE, graph)
    print(f"✓ Cached graph ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")


def compute_files_hash(files: list[tuple[str, str]]) -> str:
    """Compute hash of files for graph caching."""
    files_str = json.dumps(files, sort_keys=True)
    return hashlib.md5(files_str.encode()).hexdigest()


def _load_json_cache(cache_file: Path) -> dict:
    """Load JSON cache file, return empty dict on error."""
    if not cache_file.exists():
        return {}
    try:
        return json.load(open(cache_file))
    except json.JSONDecodeError:
        return {}


def _save_json_cache(cache_file: Path, data: dict):
    """Save dict to JSON cache file."""
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def get_cached_agent_response(question: str) -> tuple[str, list[str]] | None:
    """
    Retrieve cached agent response for a question (Local Embedding Cache).

    Returns:
        Tuple of (answer, contexts) if cached, None otherwise
    """
    cache = _load_json_cache(AGENT_RESPONSES_CACHE_FILE)
    if question in cache:
        cached = cache[question]
        return cached["answer"], cached["contexts"]
    return None


def cache_agent_response(question: str, answer: str, contexts: list[str]):
    """Store agent response in local cache for future runs."""
    cache = _load_json_cache(AGENT_RESPONSES_CACHE_FILE)
    cache[question] = {"answer": answer, "contexts": contexts}
    _save_json_cache(AGENT_RESPONSES_CACHE_FILE, cache)


def compute_samples_hash(samples: list) -> str:
    """Compute hash of samples for RAGAS result caching."""
    samples_str = json.dumps(
        [
            {
                "user_input": s.user_input,
                "response": s.response,
                "retrieved_contexts": s.retrieved_contexts,
                "reference": s.reference,
            }
            for s in samples
        ],
        sort_keys=True,
    )
    return hashlib.md5(samples_str.encode()).hexdigest()


def get_cached_ragas_results(samples: list) -> object | None:
    """
    Retrieve cached RAGAS evaluation results (Result Caching).

    Results are cached by content hash of samples to avoid re-evaluation
    when samples haven't changed.
    """
    samples_hash = compute_samples_hash(samples)
    cache = _load_pickle_cache(RAGAS_RESULTS_CACHE_FILE)
    if cache and cache.get("hash") == samples_hash:
        print(f"✓ Loaded cached RAGAS results (hash: {samples_hash[:8]}...)")
        return cache["results"]
    return None


def cache_ragas_results(samples: list, results: object):
    """Store RAGAS evaluation results in cache indexed by samples hash."""
    samples_hash = compute_samples_hash(samples)
    _save_pickle_cache(RAGAS_RESULTS_CACHE_FILE, {"hash": samples_hash, "results": results})
    print(f"✓ Cached RAGAS results (hash: {samples_hash[:8]}...)")


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------


def load_codebase(root: Path) -> list[tuple[str, str]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Codebase path does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append((str(path.relative_to(root)), path.read_text(encoding="utf-8")))
    return files


# ---------------------------------------------------------------------------
# 2. EXTRACT RELATIONSHIPS & BUILD GRAPH
# ---------------------------------------------------------------------------


def extract_relationships(relationship_extractor, files: list[tuple[str, str]]) -> list[CodeRelationship]:
    relationships = []
    for rel_path, content in files:
        result = relationship_extractor.invoke(
            {"messages": [{"role": "user", "content": f"File: {rel_path}\n\n{content}"}]}
        )
        relationships.extend(result["structured_response"].relationships)
    return relationships


def build_graph(relationships: list[CodeRelationship]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for r in relationships:
        graph.add_edge(r.subject.strip(), r.obj.strip(), relation=r.predicate)
    return graph


# ---------------------------------------------------------------------------
# 3. RETRIEVE
# ---------------------------------------------------------------------------


def match_nodes(graph: nx.DiGraph, entity: str) -> list[str]:
    needle = entity.strip().lower()
    return [node for node in graph.nodes() if needle in node.lower() or node.lower() in needle]


def graph_retrieve(graph: nx.DiGraph, entity_extractor, query: str, depth: int = 2) -> str:
    result = entity_extractor.invoke({"messages": [{"role": "user", "content": query}]})
    entities = result["structured_response"].names

    relationships = []
    for entity in entities:
        for node in match_nodes(graph, entity):
            neighbourhood = nx.ego_graph(graph, node, radius=depth, undirected=True)
            for source, target, data in neighbourhood.edges(data=True):
                relationships.append(f"{source} -[{data['relation']}]-> {target}")

    if not relationships:
        return "No relevant graph data found."
    return "Knowledge Graph context:\n" + "\n".join(sorted(set(relationships)))


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------


def _create_agent(llm, response_format, system_prompt: str):
    """Helper to create a structured agent with consistent configuration."""
    return create_agent(
        model=llm,
        tools=[],
        response_format=response_format,
        system_prompt=system_prompt,
    )


def run_query(qa_agent, entity_extractor, graph: nx.DiGraph, question: str, depth: int = 2) -> tuple[str, list[str]]:
    # Check local cache first before running agent
    cached = get_cached_agent_response(question)
    if cached:
        answer, contexts = cached
        print("✓ Loaded cached response")
        return answer, contexts

    context = graph_retrieve(graph, entity_extractor, question, depth=depth)
    result = qa_agent.invoke(
        {"messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]}
    )
    answer = result["messages"][-1].content

    # Cache this response for future runs
    cache_agent_response(question, answer, [context])
    return answer, [context]


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------


from golden_dataset import TEST_CASES


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_CODEBASE)
    parser.add_argument("--depth", type=int, default=2, help="Graph neighbourhood radius (default: 2)")
    args = parser.parse_args()

    files = load_codebase(args.repo.resolve())
    if not files:
        raise SystemExit(f"No source files found under {args.repo}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    relationship_extractor = _create_agent(
        llm,
        GraphDocument,
        "Extract code relationships from the source file as (subject, predicate, object) facts.\n"
        "Use ALL_CAPS predicates such as: DEFINES, IMPORTS, USES, CALLS, DEPENDS_ON, "
        "INHERITS_FROM, IMPLEMENTS, VALIDATES, SENDS_TO, CONFIGURES.\n"
        "Subjects and objects should be class names, function names, module paths, or file paths.\n"
        "Capture imports, constructor dependencies, method calls, and cross-module relationships.\n"
        "Be consistent: use the same name for the same class or module across relationships.",
    )
    entity_extractor = _create_agent(
        llm,
        Entities,
        "Extract code-related entities from the user message: class names, function names, module names, and file paths.",
    )
    qa_agent = _create_agent(
        llm,
        None,
        "You are a codebase assistant. Answer ONLY from the context provided in the user message. "
        "Reference specific classes, files, and relationships when possible. "
        "If you cannot answer from context, say so.",
    )

    # ============ STAGE 1: BUILD/LOAD GRAPH ============
    print(f"Loaded {len(files)} files – extracting knowledge graph...")
    files_hash = compute_files_hash(files)
    graph = get_cached_graph(files_hash)

    if graph is None:
        relationships = extract_relationships(relationship_extractor, files)
        graph = build_graph(relationships)
        cache_graph(files_hash, graph, relationships)

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ============ STAGE 2: GENERATE EVALUATION SAMPLES ============
    print("\nRunning agent on test cases...")
    samples = []
    for tc in TEST_CASES:
        answer, contexts = run_query(qa_agent, entity_extractor, graph, tc["question"], depth=args.depth)
        samples.append(SingleTurnSample(
            user_input=tc["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=tc["reference"],
        ))

        print(f"\nQ: {tc['question']}")
        print("Context retrieved:")
        print(f"  {contexts[0][:200]}{'...' if len(contexts[0]) > 200 else ''}")
        print(f"Answer: {answer}")

    # ============ STAGE 3: RUN RAGAS EVALUATION ============
    print("\nRunning RAGAS evaluation (GPT-4o-mini as judge)...")

    # Check if RAGAS results are cached first
    cached_results = get_cached_ragas_results(samples)
    if cached_results is not None:
        results = cached_results
    else:
        from openai import OpenAI
        openai_client = OpenAI()
        evaluator_llm = llm_factory("gpt-4o-mini", client=openai_client)
        lc_embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")
        eval_dataset = EvaluationDataset(samples=samples)

        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                LLMContextPrecisionWithReference(),
                LLMContextRecall(),
                Faithfulness(),
                AnswerRelevancy(embeddings=lc_embeddings),
                FactualCorrectness(),
            ],
            llm=evaluator_llm,
        )
        cache_ragas_results(samples, results)

    # ============ STAGE 4: DISPLAY RESULTS ============
    def _get_status_icon(val: float) -> str:
        """Get status icon based on metric value threshold."""
        if val >= 0.7:
            return "✅"
        if val >= 0.5:
            return "⚠️ "
        return "❌"

    def _format_metric_value(val: float) -> str:
        """Format metric value with status icon."""
        icon = _get_status_icon(val)
        return f"{icon} {val:.3f}"

    df = results.to_pandas()
    metric_cols = [
        "llm_context_precision_with_reference",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "factual_correctness",
    ]

    print("\nRAGAS SCORECARD - GraphRAG on sample_project")
    print("Judge: gpt-4o-mini (via OpenAI) | Caching: Enabled")
    print("\nPer-question breakdown:")
    for _, row in df.iterrows():
        print(f"\n  Q: {row['user_input'][:65]}")
        for col in metric_cols:
            if col in row:
                val = row[col]
                print(f"    {_format_metric_value(val)} {col:<42}: {val:.3f}")

    print("\nAggregate averages:")
    for col in metric_cols:
        if col in df.columns:
            avg = df[col].mean()
            print(f"  {_format_metric_value(avg)} {col:<42}: {avg:.3f}")
    print()
