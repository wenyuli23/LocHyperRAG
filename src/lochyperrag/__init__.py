"""LoCHyperRAG: local, deterministic HyperGraphRAG indexing and retrieval."""

from .core import build_hypergraph_index, query_hypergraph_index

build_index = build_hypergraph_index
query_index = query_hypergraph_index

__all__ = ["build_index", "query_index", "build_hypergraph_index", "query_hypergraph_index"]
__version__ = "0.1.0"
