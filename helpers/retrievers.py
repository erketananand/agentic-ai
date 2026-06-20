# Retrievers Module
# Custom retriever implementations for different search strategies

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi


class BM25Retriever(BaseRetriever):
    """
    LangChain-compatible retriever using rank_bm25 (BM25 algorithm).
    Ranks documents based on keyword matching and term frequency.
    No embeddings needed — fast and lightweight.

    BM25 is effective for lexical (keyword-based) search and works well for:
    - Finding exact keyword matches
    - Fast retrieval without GPU embeddings
    - Codebases with consistent terminology

    Attributes:
        docs: List of document chunks to search over
        bm25: BM25Okapi instance for scoring
        k: Number of top documents to retrieve (default: 4)

    Example:
        # Create a BM25 retriever from chunks
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        bm25_obj = BM25Okapi(tokenized)
        retriever = BM25Retriever(docs=chunks, bm25=bm25_obj)

        # Use in a LangChain agent
        retriever_tool = create_retriever_tool(retriever, name="search_codebase", ...)
    """
    docs: list  # List of document chunks
    bm25: object  # BM25Okapi instance for scoring
    k: int = 4  # Number of top documents to retrieve

    class Config:
        # Allow arbitrary types (needed for bm25 object)
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list:
        """
        Retrieve the k most relevant documents for a query using BM25 scoring.

        Args:
            query: The search query string
            run_manager: Callback manager for tracking retrieval

        Returns:
            List of top-k most relevant Document objects
        """
        # Tokenize query into lowercase words
        tokens = query.lower().split()
        # Get BM25 scores for each document
        scores = self.bm25.get_scores(tokens)
        # Find indices of the top-k highest scoring documents
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        # Return the corresponding documents
        return [self.docs[i] for i in top_indices]
