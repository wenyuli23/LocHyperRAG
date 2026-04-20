"""Hypergraph-based indexing and retrieval for document-grounded entity graphs."""

from __future__ import annotations

import json
import math
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import yaml
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

try:
    import igraph as ig
    import leidenalg
except ImportError:  # pragma: no cover
    ig = None
    leidenalg = None


ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "LOC",
    "EVENT",
    "FAC",
    "NORP",
    "WORK_OF_ART",
}
EDGE_TYPE_WEIGHTS = {"document": 1.0, "span": 2.0}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_EMBED_TEXT = 1024
_NLP = None
_EMBEDDER = None


@dataclass
class ResolvedProjectPaths:
    """Resolved project directories for input and output."""

    root: Path
    input_dir: Path
    output_dir: Path


def build_hypergraph_index(
    root: str | Path,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_docs: int | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Build a document-and-span weighted hypergraph index."""
    paths = resolve_project_paths(root=root, input_dir=input_dir, output_dir=output_dir)
    documents = load_documents(paths.input_dir, max_docs=max_docs)
    nlp = load_nlp()

    alias_to_entity_id: dict[str, str] = {}
    entity_state: dict[str, dict[str, Any]] = {}
    alias_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    hyperedge_rows: list[dict[str, Any]] = []
    incidence_rows: list[dict[str, Any]] = []
    document_text_unit_ids: dict[str, list[str]] = defaultdict(list)
    entity_text_units: dict[str, set[str]] = defaultdict(set)
    entity_documents: dict[str, set[str]] = defaultdict(set)
    next_entity_hid = 1
    next_span_hid = 1
    next_hyperedge_hid = 1

    for _, document in documents.iterrows():
        doc_id = str(document["id"])
        doc_text = str(document["text"])
        spacy_doc = nlp(doc_text)
        mentions, alias_rows, next_entity_hid = extract_mentions(
            spacy_doc=spacy_doc,
            document_id=doc_id,
            alias_to_entity_id=alias_to_entity_id,
            entity_state=entity_state,
            alias_rows=alias_rows,
            next_entity_hid=next_entity_hid,
        )

        doc_hyperedge_id = stable_id("hyperedge", f"document:{doc_id}")
        doc_hyperedge_entities = sorted({mention["entity_id"] for mention in mentions})
        incidence_rows.extend(
            create_incidence_rows(
                hyperedge_id=doc_hyperedge_id,
                span_id=None,
                document_id=doc_id,
                mentions=mentions,
            )
        )
        hyperedge_rows.append({
            "hyperedge_id": doc_hyperedge_id,
            "human_readable_id": next_hyperedge_hid,
            "type": "document",
            "document_id": doc_id,
            "span_id": None,
            "weight": 0.0,
            "text": doc_text,
            "topic_id": None,
            "community": None,
            "entity_ids": doc_hyperedge_entities,
        })
        next_hyperedge_hid += 1

        for entity_id in doc_hyperedge_entities:
            entity_documents[entity_id].add(doc_id)

        for sent in spacy_doc.sents:
            span_mentions = [
                mention
                for mention in mentions
                if sent.start_char <= mention["start"] and mention["end"] <= sent.end_char
            ]
            unique_entities = sorted({mention["entity_id"] for mention in span_mentions})
            if not unique_entities:
                continue

            span_id = stable_id(
                "span",
                f"{doc_id}:{sent.start_char}:{sent.end_char}:{sent.text.strip()}",
            )
            span_rows.append({
                "span_id": span_id,
                "human_readable_id": next_span_hid,
                "document_id": doc_id,
                "start": sent.start_char,
                "end": sent.end_char,
                "text": sent.text.strip(),
            })
            document_text_unit_ids[doc_id].append(span_id)
            next_span_hid += 1

            span_hyperedge_id = stable_id("hyperedge", f"span:{span_id}")
            incidence_rows.extend(
                create_incidence_rows(
                    hyperedge_id=span_hyperedge_id,
                    span_id=span_id,
                    document_id=doc_id,
                    mentions=span_mentions,
                )
            )
            hyperedge_rows.append({
                "hyperedge_id": span_hyperedge_id,
                "human_readable_id": next_hyperedge_hid,
                "type": "span",
                "document_id": doc_id,
                "span_id": span_id,
                "weight": 0.0,
                "text": sent.text.strip(),
                "topic_id": None,
                "community": None,
                "entity_ids": unique_entities,
            })
            next_hyperedge_hid += 1

            for entity_id in unique_entities:
                entity_text_units[entity_id].add(span_id)
                entity_documents[entity_id].add(doc_id)

    hyperedges_df = pd.DataFrame(hyperedge_rows)
    incidence_df = pd.DataFrame(incidence_rows)
    evidence_spans_df = pd.DataFrame(span_rows)
    aliases_df = pd.DataFrame(alias_rows).drop_duplicates(
        subset=["alias", "canonical_name", "entity_id"]
    )

    hyperedges_df = apply_hyperedge_weights(hyperedges_df, incidence_df)
    (
        projected_relationships_df,
        graph,
        relationship_spans,
    ) = project_hypergraph(hyperedges_df, incidence_df, entity_state)
    community_memberships = detect_communities(graph)
    hyperedges_df["community"] = hyperedges_df["entity_ids"].apply(
        lambda entity_ids: dominant_community(entity_ids, community_memberships)
    )

    entities_df = finalize_entities(
        entity_state=entity_state,
        entity_text_units=entity_text_units,
        entity_documents=entity_documents,
        graph=graph,
    )
    text_units_df = create_text_units(
        evidence_spans_df=evidence_spans_df,
        incidence_df=incidence_df,
        relationship_spans=relationship_spans,
    )
    documents_df = finalize_documents(
        documents=documents,
        document_text_unit_ids=document_text_unit_ids,
    )
    communities_df = create_communities(
        community_memberships=community_memberships,
        entities_df=entities_df,
        relationships_df=projected_relationships_df,
        incidence_df=incidence_df,
    )
    community_reports_df = create_community_reports(
        communities_df=communities_df,
        entities_df=entities_df,
        relationships_df=projected_relationships_df,
        hyperedges_df=hyperedges_df,
        evidence_spans_df=evidence_spans_df,
    )
    entity_embeddings_df, hyperedge_embeddings_df = build_embeddings(
        output_dir=paths.output_dir,
        entities_df=entities_df,
        hyperedges_df=hyperedges_df,
        incidence_df=incidence_df,
        embedding_model_name=embedding_model_name,
    )

    relationships_df = projected_relationships_df[
        [
            "id",
            "human_readable_id",
            "source",
            "target",
            "description",
            "weight",
            "combined_degree",
            "text_unit_ids",
        ]
    ].copy()

    write_outputs(
        output_dir=paths.output_dir,
        documents_df=documents_df,
        entities_df=entities_df,
        relationships_df=relationships_df,
        projected_relationships_df=projected_relationships_df,
        communities_df=communities_df,
        community_reports_df=community_reports_df,
        text_units_df=text_units_df,
        evidence_spans_df=evidence_spans_df,
        hyperedges_df=hyperedges_df,
        incidence_df=incidence_df,
        aliases_df=aliases_df,
        entity_embeddings_df=entity_embeddings_df,
        hyperedge_embeddings_df=hyperedge_embeddings_df,
        graph=graph,
    )

    stats = {
        "documents": len(documents_df),
        "entities": len(entities_df),
        "hyperedges": len(hyperedges_df),
        "evidence_spans": len(evidence_spans_df),
        "relationships": len(projected_relationships_df),
        "communities": len(communities_df),
        "output_dir": str(paths.output_dir),
    }
    (paths.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return stats


def query_hypergraph_index(
    root: str | Path,
    data_dir: str | Path | None,
    query: str,
    top_k_hyperedges: int = 8,
    top_k_entities: int = 10,
    top_k_communities: int = 3,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Query a hypergraph index and build a local/global context pack."""
    paths = resolve_project_paths(root=root, output_dir=data_dir)
    output_dir = paths.output_dir

    hyperedges_df = pd.read_parquet(output_dir / "hyperedges.parquet")
    incidence_df = pd.read_parquet(output_dir / "incidence.parquet")
    entities_df = pd.read_parquet(output_dir / "entities.parquet")
    communities_df = pd.read_parquet(output_dir / "communities.parquet")
    community_reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
    evidence_spans_df = pd.read_parquet(output_dir / "evidence_spans.parquet")
    entity_vectors = np.load(output_dir / "embeddings" / "entity_final.npy")
    hyperedge_vectors = np.load(output_dir / "embeddings" / "hyperedge_final.npy")

    embedder = load_embedder(embedding_model_name=embedding_model_name)
    query_vector = normalize_vectors(embedder.encode([query], show_progress_bar=False))[0]
    entity_query_vector = align_query_vector(query_vector, entity_vectors.shape[1])
    hyperedge_query_vector = align_query_vector(query_vector, hyperedge_vectors.shape[1])

    top_hyperedges = score_hyperedges(
        hyperedges_df,
        hyperedge_vectors,
        hyperedge_query_vector,
    ).head(
        top_k_hyperedges
    )
    selected_hyperedge_ids = top_hyperedges["hyperedge_id"].tolist()
    local_spans = top_hyperedges[top_hyperedges["type"] == "span"].copy()
    local_documents = top_hyperedges[top_hyperedges["type"] == "document"].copy()
    entity_scores = score_entities(
        selected_hyperedge_ids=selected_hyperedge_ids,
        incidence_df=incidence_df,
        entities_df=entities_df,
        entity_vectors=entity_vectors,
        query_vector=entity_query_vector,
        hyperedge_scores=top_hyperedges,
    ).head(top_k_entities)
    community_scores = score_communities(
        selected_hyperedge_ids=selected_hyperedge_ids,
        top_hyperedges=top_hyperedges,
        communities_df=communities_df,
        community_reports_df=community_reports_df,
        incidence_df=incidence_df,
    ).head(top_k_communities)

    supporting_spans = evidence_spans_df[
        evidence_spans_df["span_id"].isin(local_spans["span_id"].dropna().tolist())
    ].copy()

    response = render_query_response(
        query=query,
        local_spans=local_spans,
        local_documents=local_documents,
        entity_scores=entity_scores,
        community_scores=community_scores,
        supporting_spans=supporting_spans,
    )
    return {
        "response": response,
        "local_hyperedges": top_hyperedges.to_dict(orient="records"),
        "top_entities": entity_scores.to_dict(orient="records"),
        "top_communities": community_scores.to_dict(orient="records"),
    }


def resolve_project_paths(
    root: str | Path,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ResolvedProjectPaths:
    """Resolve GraphRAG-style project paths from a project root."""
    root_path = Path(root).resolve()
    settings_path = root_path / "settings.yaml"
    settings = {}
    if settings_path.exists():
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8-sig")) or {}

    resolved_input = (
        Path(input_dir).resolve()
        if input_dir is not None
        else root_path / settings.get("input", {}).get("storage", {}).get("base_dir", "input")
    )
    resolved_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root_path / settings.get("output", {}).get("base_dir", "output")
    )
    return ResolvedProjectPaths(
        root=root_path,
        input_dir=resolved_input.resolve(),
        output_dir=resolved_output.resolve(),
    )


def load_documents(input_dir: Path, max_docs: int | None) -> pd.DataFrame:
    """Load text documents from a GraphRAG-style input directory."""
    paths = sorted(input_dir.rglob("*.txt"))
    if max_docs is not None:
        paths = paths[:max_docs]

    rows = []
    for index, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8-sig").strip()
        document_id = stable_id("document", f"{path.name}:{text}")
        rows.append({
            "id": document_id,
            "human_readable_id": index,
            "title": path.name,
            "text": text,
            "creation_date": datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(
                "%Y-%m-%d %H:%M:%S %z"
            ),
            "metadata": None,
        })

    return pd.DataFrame(rows)


def load_nlp():
    """Load the English spaCy pipeline once."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    return _NLP


def load_embedder(embedding_model_name: str):
    """Load the sentence-transformer once."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            _EMBEDDER = SentenceTransformer(embedding_model_name)
        except Exception:
            _EMBEDDER = SentenceTransformer(embedding_model_name, local_files_only=True)
    return _EMBEDDER


def extract_mentions(
    spacy_doc,
    document_id: str,
    alias_to_entity_id: dict[str, str],
    entity_state: dict[str, dict[str, Any]],
    alias_rows: list[dict[str, Any]],
    next_entity_hid: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Extract and canonicalize entity mentions from a document."""
    mentions: list[dict[str, Any]] = []
    for ent in spacy_doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue
        canonical_name = normalize_alias(ent.text)
        if not canonical_name:
            continue

        entity_id = alias_to_entity_id.get(canonical_name)
        if entity_id is None:
            entity_id = stable_id("entity", canonical_name)
            alias_to_entity_id[canonical_name] = entity_id
            entity_state[entity_id] = {
                "entity_id": entity_id,
                "human_readable_id": next_entity_hid,
                "title": ent.text.strip(),
                "type": ent.label_,
                "description": f"{ent.text.strip()} ({ent.label_}) observed in support ticket evidence.",
                "frequency": 0,
                "aliases": Counter(),
            }
            next_entity_hid += 1

        entity_state[entity_id]["frequency"] += 1
        entity_state[entity_id]["aliases"][ent.text.strip()] += 1
        alias_rows.append({
            "alias": ent.text.strip(),
            "canonical_name": canonical_name,
            "entity_id": entity_id,
            "document_id": document_id,
        })
        mentions.append({
            "document_id": document_id,
            "entity_id": entity_id,
            "text": ent.text.strip(),
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": 1.0,
        })

    for state in entity_state.values():
        if state["aliases"]:
            state["title"] = state["aliases"].most_common(1)[0][0]

    return mentions, alias_rows, next_entity_hid


def create_incidence_rows(
    hyperedge_id: str,
    span_id: str | None,
    document_id: str,
    mentions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate mentions into hyperedge incidence rows."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for mention in mentions:
        entity_id = mention["entity_id"]
        key = (hyperedge_id, entity_id)
        row = grouped.get(key)
        if row is None:
            row = {
                "hyperedge_id": hyperedge_id,
                "entity_id": entity_id,
                "role": mention["label"],
                "mention_count": 0,
                "confidence": 0.0,
                "document_id": document_id,
                "span_id": span_id,
            }
            grouped[key] = row
        row["mention_count"] += 1
        row["confidence"] = max(row["confidence"], mention["confidence"])
    return list(grouped.values())


def apply_hyperedge_weights(
    hyperedges_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weighted hyperedge strengths."""
    if hyperedges_df.empty:
        return hyperedges_df

    incidence_stats = (
        incidence_df.groupby("hyperedge_id")
        .agg(
            edge_size=("entity_id", "nunique"),
            mention_count=("mention_count", "sum"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    weighted = hyperedges_df.merge(incidence_stats, on="hyperedge_id", how="left")
    weighted["edge_size"] = weighted["edge_size"].fillna(0)
    weighted["mention_count"] = weighted["mention_count"].fillna(0)
    weighted["mean_confidence"] = weighted["mean_confidence"].fillna(0.0)
    weighted["within_span_density"] = weighted.apply(
        lambda row: 0.0
        if row["edge_size"] == 0
        else row["mention_count"] / row["edge_size"],
        axis=1,
    )
    weighted["weight"] = weighted.apply(calculate_hyperedge_weight, axis=1)
    return weighted.drop(
        columns=["edge_size", "mention_count", "mean_confidence", "within_span_density"]
    )


def calculate_hyperedge_weight(row: pd.Series) -> float:
    """Apply the v1 hyperedge weighting formula."""
    edge_size = int(row.get("edge_size", 0))
    if edge_size == 0:
        return 0.0
    mention_count = max(float(row.get("mention_count", 0)), 1.0)
    mean_confidence = float(row.get("mean_confidence", 0.0))
    density = max(float(row.get("within_span_density", 0.0)), 1.0)
    base = EDGE_TYPE_WEIGHTS.get(str(row["type"]), 1.0)
    return round(
        base
        * mean_confidence
        * density
        * math.log1p(mention_count)
        / math.log(2 + edge_size),
        6,
    )


def project_hypergraph(
    hyperedges_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    entity_state: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, nx.Graph, dict[str, list[str]]]:
    """Create a weighted 2-section projection from the hypergraph."""
    graph = nx.Graph()
    for entity_id, state in entity_state.items():
        graph.add_node(entity_id, title=state["title"], frequency=state["frequency"])

    pair_state: dict[tuple[str, str], dict[str, Any]] = {}
    relationship_spans: dict[str, list[str]] = defaultdict(list)
    entity_titles = {entity_id: state["title"] for entity_id, state in entity_state.items()}

    hyperedge_lookup = hyperedges_df.set_index("hyperedge_id").to_dict(orient="index")
    incidence_by_hyperedge = defaultdict(list)
    for row in incidence_df.to_dict(orient="records"):
        incidence_by_hyperedge[row["hyperedge_id"]].append(row)

    for hyperedge_id, rows in incidence_by_hyperedge.items():
        unique_entities = sorted({row["entity_id"] for row in rows})
        hyperedge = hyperedge_lookup[hyperedge_id]
        weight = float(hyperedge.get("weight", 0.0))
        if len(unique_entities) < 2:
            continue
        contribution = weight / max(len(unique_entities) - 1, 1)
        for left_id, right_id in combinations(unique_entities, 2):
            source_id, target_id = sorted((left_id, right_id))
            key = (source_id, target_id)
            state = pair_state.get(key)
            if state is None:
                relationship_id = stable_id("relationship", f"{source_id}:{target_id}")
                state = {
                    "id": relationship_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "source": entity_titles[source_id],
                    "target": entity_titles[target_id],
                    "description": "Entities co-mentioned across shared document and evidence-span hyperedges.",
                    "weight": 0.0,
                    "shared_hyperedge_ids": [],
                    "text_unit_ids": [],
                }
                pair_state[key] = state
            state["weight"] += contribution
            state["shared_hyperedge_ids"].append(hyperedge_id)
            if has_value(hyperedge.get("span_id")):
                state["text_unit_ids"].append(str(hyperedge["span_id"]))
                relationship_spans[state["id"]].append(str(hyperedge["span_id"]))

    relationship_rows = []
    for index, state in enumerate(
        sorted(pair_state.values(), key=lambda item: (-item["weight"], item["source"], item["target"])),
        start=1,
    ):
        graph.add_edge(state["source_id"], state["target_id"], weight=state["weight"])
        relationship_rows.append({
            "id": state["id"],
            "human_readable_id": index,
            "source": state["source"],
            "target": state["target"],
            "description": state["description"],
            "weight": round(state["weight"], 6),
            "combined_degree": 0,
            "text_unit_ids": sorted(set(state["text_unit_ids"])),
            "shared_hyperedge_ids": sorted(set(state["shared_hyperedge_ids"])),
        })

    for row in relationship_rows:
        source_id = stable_id("entity", normalize_alias(row["source"]))
        target_id = stable_id("entity", normalize_alias(row["target"]))
        row["combined_degree"] = graph.degree(source_id) + graph.degree(target_id)

    return pd.DataFrame(relationship_rows), graph, relationship_spans


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Detect communities with Leiden when available, otherwise fall back to Louvain."""
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {node_id: index for index, node_id in enumerate(graph.nodes())}

    if ig is not None and leidenalg is not None:
        nodes = list(graph.nodes())
        node_index = {node_id: index for index, node_id in enumerate(nodes)}
        igraph_graph = ig.Graph()
        igraph_graph.add_vertices(len(nodes))
        igraph_graph.add_edges(
            [(node_index[left], node_index[right]) for left, right in graph.edges()]
        )
        igraph_graph.es["weight"] = [
            float(graph.edges[left, right].get("weight", 1.0))
            for left, right in graph.edges()
        ]
        partition = leidenalg.find_partition(
            igraph_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=igraph_graph.es["weight"],
            seed=42,
        )
        memberships = {}
        for community_id, vertex_ids in enumerate(partition):
            for vertex_id in vertex_ids:
                memberships[nodes[vertex_id]] = community_id
        return memberships

    communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
    memberships = {}
    for community_id, members in enumerate(communities):
        for node_id in members:
            memberships[node_id] = community_id
    return memberships


def finalize_entities(
    entity_state: dict[str, dict[str, Any]],
    entity_text_units: dict[str, set[str]],
    entity_documents: dict[str, set[str]],
    graph: nx.Graph,
) -> pd.DataFrame:
    """Create GraphRAG-compatible entities output."""
    rows = []
    for state in sorted(entity_state.values(), key=lambda item: item["human_readable_id"]):
        entity_id = state["entity_id"]
        rows.append({
            "id": entity_id,
            "human_readable_id": state["human_readable_id"],
            "title": state["title"],
            "type": state["type"],
            "description": state["description"],
            "text_unit_ids": sorted(entity_text_units.get(entity_id, set())),
            "frequency": state["frequency"],
            "degree": graph.degree(entity_id) if graph.has_node(entity_id) else 0,
            "x": 0,
            "y": 0,
            "document_ids": sorted(entity_documents.get(entity_id, set())),
        })
    return pd.DataFrame(rows)


def create_text_units(
    evidence_spans_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    relationship_spans: dict[str, list[str]],
) -> pd.DataFrame:
    """Create GraphRAG-compatible text_units output from evidence spans."""
    entity_ids_by_span = defaultdict(set)
    for row in incidence_df.to_dict(orient="records"):
        span_id = row.get("span_id")
        if has_value(span_id):
            entity_ids_by_span[str(span_id)].add(str(row["entity_id"]))

    relationship_ids_by_span = defaultdict(set)
    for relationship_id, span_ids in relationship_spans.items():
        for span_id in span_ids:
            relationship_ids_by_span[str(span_id)].add(relationship_id)

    rows = []
    for _, row in evidence_spans_df.iterrows():
        span_id = str(row["span_id"])
        rows.append({
            "id": span_id,
            "human_readable_id": int(row["human_readable_id"]),
            "text": row["text"],
            "n_tokens": len(str(row["text"]).split()),
            "document_ids": [str(row["document_id"])],
            "entity_ids": sorted(entity_ids_by_span.get(span_id, set())),
            "relationship_ids": sorted(relationship_ids_by_span.get(span_id, set())),
            "covariate_ids": [],
        })
    return pd.DataFrame(rows)


def finalize_documents(
    documents: pd.DataFrame,
    document_text_unit_ids: dict[str, list[str]],
) -> pd.DataFrame:
    """Attach evidence spans to documents output."""
    rows = []
    for _, row in documents.iterrows():
        rows.append({
            "id": row["id"],
            "human_readable_id": row["human_readable_id"],
            "title": row["title"],
            "text": row["text"],
            "text_unit_ids": document_text_unit_ids.get(str(row["id"]), []),
            "creation_date": row["creation_date"],
            "metadata": row["metadata"],
        })
    return pd.DataFrame(rows)


def create_communities(
    community_memberships: dict[str, int],
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build GraphRAG-compatible community rows."""
    grouped_entities = defaultdict(list)
    for entity_id, community_id in community_memberships.items():
        grouped_entities[community_id].append(entity_id)

    relationship_lookup = relationships_df.to_dict(orient="records")
    incidence_by_entity = defaultdict(set)
    for row in incidence_df.to_dict(orient="records"):
        if has_value(row.get("span_id")):
            incidence_by_entity[str(row["entity_id"])].add(str(row["span_id"]))

    rows = []
    for index, community_id in enumerate(sorted(grouped_entities), start=1):
        entity_ids = sorted(grouped_entities[community_id])
        entity_titles = entities_df[entities_df["id"].isin(entity_ids)]["title"].tolist()
        relationship_ids = [
            row["id"]
            for row in relationship_lookup
            if stable_id("entity", normalize_alias(row["source"])) in entity_ids
            and stable_id("entity", normalize_alias(row["target"])) in entity_ids
        ]
        text_unit_ids = sorted(
            {
                span_id
                for entity_id in entity_ids
                for span_id in incidence_by_entity.get(entity_id, set())
            }
        )
        rows.append({
            "id": stable_id("community", str(community_id)),
            "human_readable_id": index,
            "community": community_id,
            "level": 0,
            "parent": -1,
            "children": [],
            "title": f"Community {community_id}: {', '.join(entity_titles[:3])}",
            "entity_ids": entity_ids,
            "relationship_ids": relationship_ids,
            "text_unit_ids": text_unit_ids,
            "period": datetime.now(UTC).strftime("%Y-%m-%d"),
            "size": len(entity_ids),
        })
    return pd.DataFrame(rows)


def create_community_reports(
    communities_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    hyperedges_df: pd.DataFrame,
    evidence_spans_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic GraphRAG-style community reports."""
    entity_lookup = entities_df.set_index("id").to_dict(orient="index")
    relationships = relationships_df.to_dict(orient="records")
    hyperedges = hyperedges_df.to_dict(orient="records")
    span_lookup = evidence_spans_df.set_index("span_id").to_dict(orient="index")

    rows = []
    for _, community in communities_df.iterrows():
        entity_ids = list(community["entity_ids"])
        top_entities = sorted(
            (entity_lookup[entity_id] for entity_id in entity_ids),
            key=lambda item: (-item["frequency"], item["title"]),
        )
        community_relationships = [
            row
            for row in relationships
            if stable_id("entity", normalize_alias(row["source"])) in entity_ids
            and stable_id("entity", normalize_alias(row["target"])) in entity_ids
        ]
        community_hyperedges = [
            row
            for row in hyperedges
            if row.get("community") == int(community["community"])
        ]
        supporting_span_ids = [
            str(row["span_id"])
            for row in community_hyperedges
            if has_value(row.get("span_id"))
        ]
        supporting_spans = [
            span_lookup[span_id]["text"]
            for span_id in supporting_span_ids
            if span_id in span_lookup
        ][:3]

        summary = (
            f"This community groups {len(entity_ids)} entities that repeatedly co-occur "
            f"across {len({row['document_id'] for row in community_hyperedges})} documents "
            f"and {len(set(supporting_span_ids))} evidence spans."
        )
        strongest_relationships = [
            f"{row['source']} <-> {row['target']}"
            for row in community_relationships[:3]
        ]
        findings = [
            {
                "summary": f"Top entities: {', '.join(item['title'] for item in top_entities[:3])}",
                "explanation": "These entities have the strongest local support within shared tickets and evidence spans.",
            },
            {
                "summary": f"Strongest relationships: {', '.join(strongest_relationships) or 'None'}",
                "explanation": "These projected edges receive the largest cumulative weight from shared hyperedges.",
            },
            {
                "summary": "Representative evidence spans",
                "explanation": " | ".join(supporting_spans) if supporting_spans else "No evidence spans available.",
            },
        ]
        full_content = "\n".join(
            [
                f"# {community['title']}",
                "",
                summary,
                "",
                "## Top entities",
                ", ".join(item["title"] for item in top_entities[:5]) or "None",
                "",
                "## Strongest relationships",
                "\n".join(
                    f"- {row['source']} <-> {row['target']} (weight={row['weight']:.3f})"
                    for row in community_relationships[:5]
                )
                or "- None",
                "",
                "## Representative evidence",
                "\n".join(f"- {text}" for text in supporting_spans) or "- None",
            ]
        )
        full_content_json = json.dumps(
            {
                "title": community["title"],
                "summary": summary,
                "findings": findings,
                "rating": round(sum(row["weight"] for row in community_relationships), 3),
                "rating_explanation": "Higher ratings indicate denser within-community support from weighted document/span co-occurrence.",
            },
            indent=2,
        )
        rows.append({
            "id": stable_id("community_report", str(community["community"])),
            "human_readable_id": int(community["human_readable_id"]),
            "community": int(community["community"]),
            "level": int(community["level"]),
            "parent": int(community["parent"]),
            "children": list(community["children"]),
            "title": community["title"],
            "summary": summary,
            "full_content": full_content,
            "rank": round(sum(row["weight"] for row in community_relationships), 3),
            "rating_explanation": "Higher ranks indicate denser within-community support from weighted document/span co-occurrence.",
            "findings": findings,
            "full_content_json": full_content_json,
            "period": community["period"],
            "size": int(community["size"]),
        })
    return pd.DataFrame(rows)


def build_embeddings(
    output_dir: Path,
    entities_df: pd.DataFrame,
    hyperedges_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    embedding_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create semantic, structural, and final embeddings."""
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embedder = load_embedder(embedding_model_name=embedding_model_name)
    entity_texts = [
        truncate_embed_text(f"{row['title']}. {row['description']}")
        for _, row in entities_df.iterrows()
    ]
    hyperedge_texts = [
        truncate_embed_text(str(row["text"]))
        for _, row in hyperedges_df.iterrows()
    ]

    entity_semantic = normalize_vectors(
        embedder.encode(entity_texts, show_progress_bar=False)
    )
    hyperedge_semantic = normalize_vectors(
        embedder.encode(hyperedge_texts, show_progress_bar=False)
    )

    entity_structural, hyperedge_structural = build_structural_embeddings(
        entities_df=entities_df,
        hyperedges_df=hyperedges_df,
        incidence_df=incidence_df,
    )

    entity_final = normalize_vectors(
        np.hstack([entity_semantic, entity_structural])
        if entity_structural.size
        else entity_semantic
    )
    hyperedge_final = normalize_vectors(
        np.hstack([hyperedge_semantic, hyperedge_structural])
        if hyperedge_structural.size
        else hyperedge_semantic
    )

    np.save(embeddings_dir / "entity_semantic.npy", entity_semantic)
    np.save(embeddings_dir / "entity_structural.npy", entity_structural)
    np.save(embeddings_dir / "entity_final.npy", entity_final)
    np.save(embeddings_dir / "hyperedge_semantic.npy", hyperedge_semantic)
    np.save(embeddings_dir / "hyperedge_structural.npy", hyperedge_structural)
    np.save(embeddings_dir / "hyperedge_final.npy", hyperedge_final)

    entity_refs = [
        {
            "entity_id": row["id"],
            "semantic_vector_ref": f"embeddings/entity_semantic.npy:{index}",
            "structural_vector_ref": f"embeddings/entity_structural.npy:{index}",
            "final_vector_ref": f"embeddings/entity_final.npy:{index}",
        }
        for index, (_, row) in enumerate(entities_df.iterrows())
    ]
    hyperedge_refs = [
        {
            "hyperedge_id": row["hyperedge_id"],
            "semantic_vector_ref": f"embeddings/hyperedge_semantic.npy:{index}",
            "structural_vector_ref": f"embeddings/hyperedge_structural.npy:{index}",
            "final_vector_ref": f"embeddings/hyperedge_final.npy:{index}",
        }
        for index, (_, row) in enumerate(hyperedges_df.iterrows())
    ]
    return pd.DataFrame(entity_refs), pd.DataFrame(hyperedge_refs)


def build_structural_embeddings(
    entities_df: pd.DataFrame,
    hyperedges_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build structural embeddings from the entity-hyperedge incidence matrix."""
    if entities_df.empty or hyperedges_df.empty or incidence_df.empty:
        return np.zeros((len(entities_df), 0)), np.zeros((len(hyperedges_df), 0))

    entity_index = {entity_id: idx for idx, entity_id in enumerate(entities_df["id"].tolist())}
    hyperedge_index = {
        hyperedge_id: idx for idx, hyperedge_id in enumerate(hyperedges_df["hyperedge_id"].tolist())
    }
    row_indices = []
    col_indices = []
    data = []
    for row in incidence_df.to_dict(orient="records"):
        left = entity_index[str(row["entity_id"])]
        right = hyperedge_index[str(row["hyperedge_id"])]
        row_indices.append(left)
        col_indices.append(right)
        data.append(float(row["mention_count"]) * float(row["confidence"]))

    matrix = sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(entity_index), len(hyperedge_index)),
        dtype=np.float32,
    )

    max_components = min(32, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if max_components <= 0:
        return np.zeros((len(entities_df), 0)), np.zeros((len(hyperedges_df), 0))

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        entity_structural = normalize_vectors(svd.fit_transform(matrix))
    hyperedge_structural = matrix.T.dot(entity_structural)
    counts = np.asarray(matrix.T.sum(axis=1)).reshape(-1, 1)
    counts = np.maximum(counts, 1.0)
    hyperedge_structural = normalize_vectors(hyperedge_structural / counts)
    return entity_structural, hyperedge_structural


def write_outputs(
    output_dir: Path,
    documents_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    projected_relationships_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    community_reports_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    evidence_spans_df: pd.DataFrame,
    hyperedges_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
    aliases_df: pd.DataFrame,
    entity_embeddings_df: pd.DataFrame,
    hyperedge_embeddings_df: pd.DataFrame,
    graph: nx.Graph,
) -> None:
    """Persist all index artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_df.to_parquet(output_dir / "documents.parquet", index=False)
    entities_df.to_parquet(output_dir / "entities.parquet", index=False)
    relationships_df.to_parquet(output_dir / "relationships.parquet", index=False)
    projected_relationships_df.to_parquet(
        output_dir / "projected_relationships.parquet",
        index=False,
    )
    communities_df.to_parquet(output_dir / "communities.parquet", index=False)
    community_reports_df.to_parquet(output_dir / "community_reports.parquet", index=False)
    text_units_df.to_parquet(output_dir / "text_units.parquet", index=False)
    evidence_spans_df.to_parquet(output_dir / "evidence_spans.parquet", index=False)
    hyperedges_df.to_parquet(output_dir / "hyperedges.parquet", index=False)
    incidence_df.to_parquet(output_dir / "incidence.parquet", index=False)
    aliases_df.to_parquet(output_dir / "aliases.parquet", index=False)
    entity_embeddings_df.to_parquet(output_dir / "entity_embeddings.parquet", index=False)
    hyperedge_embeddings_df.to_parquet(output_dir / "hyperedge_embeddings.parquet", index=False)
    nx.write_graphml(graph, output_dir / "graph.graphml")


def score_hyperedges(
    hyperedges_df: pd.DataFrame,
    hyperedge_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> pd.DataFrame:
    """Rank hyperedges by semantic relevance plus local support weight."""
    if hyperedges_df.empty:
        return hyperedges_df
    sims = hyperedge_vectors @ query_vector
    scores = hyperedges_df.copy()
    scores["semantic_similarity"] = sims
    max_weight = max(float(scores["weight"].max()), 1.0)
    scores["normalized_weight"] = scores["weight"] / max_weight
    scores["type_bonus"] = scores["type"].map({"span": 1.0, "document": 0.5}).fillna(0.0)
    scores["retrieval_score"] = (
        0.75 * scores["semantic_similarity"]
        + 0.15 * scores["normalized_weight"]
        + 0.10 * scores["type_bonus"]
    )
    return scores.sort_values("retrieval_score", ascending=False).reset_index(drop=True)


def score_entities(
    selected_hyperedge_ids: list[str],
    incidence_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    entity_vectors: np.ndarray,
    query_vector: np.ndarray,
    hyperedge_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate entity relevance from selected hyperedges."""
    scores_by_hyperedge = {
        row["hyperedge_id"]: float(row["retrieval_score"])
        for row in hyperedge_scores.to_dict(orient="records")
    }
    entity_scores = defaultdict(float)
    for row in incidence_df.to_dict(orient="records"):
        hyperedge_id = str(row["hyperedge_id"])
        if hyperedge_id not in selected_hyperedge_ids:
            continue
        entity_scores[str(row["entity_id"])] += (
            scores_by_hyperedge[hyperedge_id]
            * float(row["mention_count"])
            * float(row["confidence"])
        )

    if not entity_scores:
        return pd.DataFrame(columns=["entity_id", "title", "score"])

    entity_lookup = entities_df.set_index("id").to_dict(orient="index")
    semantic_scores = entity_vectors @ query_vector
    entity_row_index = {entity_id: index for index, entity_id in enumerate(entities_df["id"].tolist())}
    rows = []
    for entity_id, local_score in entity_scores.items():
        semantic_score = float(semantic_scores[entity_row_index[entity_id]])
        total_score = 0.6 * local_score + 0.4 * semantic_score
        rows.append({
            "entity_id": entity_id,
            "title": entity_lookup[entity_id]["title"],
            "score": round(total_score, 6),
            "frequency": int(entity_lookup[entity_id]["frequency"]),
            "degree": int(entity_lookup[entity_id]["degree"]),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def score_communities(
    selected_hyperedge_ids: list[str],
    top_hyperedges: pd.DataFrame,
    communities_df: pd.DataFrame,
    community_reports_df: pd.DataFrame,
    incidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate community relevance from selected hyperedges and incident entities."""
    community_lookup = community_reports_df.set_index("community").to_dict(orient="index")
    entity_communities = {}
    for _, row in communities_df.iterrows():
        for entity_id in row["entity_ids"]:
            entity_communities[str(entity_id)] = int(row["community"])

    hyperedge_scores = {
        row["hyperedge_id"]: float(row["retrieval_score"])
        for row in top_hyperedges.to_dict(orient="records")
    }
    community_scores = defaultdict(float)
    for row in incidence_df.to_dict(orient="records"):
        hyperedge_id = str(row["hyperedge_id"])
        if hyperedge_id not in selected_hyperedge_ids:
            continue
        community_id = entity_communities.get(str(row["entity_id"]))
        if community_id is not None:
            community_scores[community_id] += hyperedge_scores[hyperedge_id]

    rows = []
    for community_id, score in community_scores.items():
        report = community_lookup.get(community_id)
        if not report:
            continue
        rows.append({
            "community": community_id,
            "title": report["title"],
            "summary": report["summary"],
            "score": round(score, 6),
            "rank": float(report["rank"]),
        })
    if not rows:
        return pd.DataFrame(columns=["community", "title", "summary", "score", "rank"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def render_query_response(
    query: str,
    local_spans: pd.DataFrame,
    local_documents: pd.DataFrame,
    entity_scores: pd.DataFrame,
    community_scores: pd.DataFrame,
    supporting_spans: pd.DataFrame,
) -> str:
    """Render a markdown response from the retrieved context."""
    lines = ["# HyperGraphRAG Results", "", f"Query: {query}", ""]

    lines.append("## Local evidence spans")
    if local_spans.empty:
        lines.append("- None")
    else:
        for _, row in local_spans.head(5).iterrows():
            lines.append(
                f"- [{row['retrieval_score']:.3f}] doc={row['document_id']} span={row['span_id']}: {str(row['text'])[:220]}"
            )

    lines.append("")
    lines.append("## Supporting documents")
    if local_documents.empty:
        lines.append("- None")
    else:
        for _, row in local_documents.head(3).iterrows():
            lines.append(
                f"- [{row['retrieval_score']:.3f}] doc={row['document_id']}: {str(row['text'])[:220]}"
            )

    lines.append("")
    lines.append("## Top entities")
    if entity_scores.empty:
        lines.append("- None")
    else:
        for _, row in entity_scores.head(10).iterrows():
            lines.append(
                f"- {row['title']} (score={row['score']:.3f}, frequency={row['frequency']}, degree={row['degree']})"
            )

    lines.append("")
    lines.append("## Top communities")
    if community_scores.empty:
        lines.append("- None")
    else:
        for _, row in community_scores.head(3).iterrows():
            lines.append(f"- {row['title']} (score={row['score']:.3f}): {row['summary']}")

    if not supporting_spans.empty:
        lines.append("")
        lines.append("## Representative support")
        for _, row in supporting_spans.head(3).iterrows():
            lines.append(f"- {str(row['text'])[:260]}")

    return "\n".join(lines)


def dominant_community(entity_ids: list[str], memberships: dict[str, int]) -> int | None:
    """Return the dominant community id for a hyperedge."""
    votes = [memberships[entity_id] for entity_id in entity_ids if entity_id in memberships]
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def normalize_alias(text: str) -> str:
    """Normalize an alias for deterministic entity merging."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\s_]+", " ", normalized)
    normalized = normalized.strip(" \t\r\n.,;:!?\"'()[]{}")
    return normalized


def has_value(value: Any) -> bool:
    """Return True when a dataframe-derived value is not null-like."""
    return not pd.isna(value)


def stable_id(prefix: str, value: str) -> str:
    """Create a deterministic UUID5 from a value."""
    return str(uuid5(NAMESPACE_URL, f"{prefix}:{value}"))


def truncate_embed_text(text: str) -> str:
    """Limit text fed into the embedding model."""
    return text[:MAX_EMBED_TEXT]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def align_query_vector(query_vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or trim a semantic query vector to match stored embedding width."""
    if len(query_vector) == target_dim:
        return query_vector
    if len(query_vector) > target_dim:
        return normalize_vectors(query_vector[:target_dim])[0]
    padding = np.zeros(target_dim - len(query_vector), dtype=np.float32)
    return normalize_vectors(np.concatenate([query_vector, padding]))[0]
