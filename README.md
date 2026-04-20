# LoCHyperRAG for Researchers

## Scope

It packages the runnable LoCHyperRAG method into a small standalone repository with:

- local indexing over plain-text documents,
- deterministic entity extraction and alias normalization,
- document-level and sentence-level hyperedge construction,
- weighted 2-section projection for graph/community analysis,
- semantic + structural embeddings for retrieval,
- a local query path that returns a short answer and supporting evidence.

It intentionally excludes the broader workspace material such as notebooks, comparison experiments, presentations, large generated outputs, and draft reporting files.

## Repository Contents

- `src/lochyperrag/core.py`: main indexing, projection, embedding, and query code.
- `src/lochyperrag/cli.py`: command-line wrapper with `doctor`, `build`, `ask`, and `demo` commands.
- `demo/input/`: small bundled corpus for smoke testing.
- `pyproject.toml`: package metadata and dependencies.
- `requirements.txt`: direct install list.
- `README.md`: beginner/student guide.
- `README_RESEARCHERS.md`: this document.

## Minimal Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m spacy download en_core_web_sm
lochyperrag doctor
lochyperrag demo
```

Or, to run on your own corpus:

```powershell
lochyperrag build --input-dir my_documents --output-dir my_output
lochyperrag ask --output-dir my_output --query "What are the main issues in this corpus?"
```

## Runtime Assumptions

The repository is designed for local execution with no API-backed extraction or generation.

Core assumptions:

- Input is a directory tree of `.txt` files.
- Each file is treated as one document.
- Entity extraction uses `spaCy` with `en_core_web_sm`.
- Semantic embeddings use `sentence-transformers/all-MiniLM-L6-v2`.
- Community detection uses Leiden if `igraph` and `leidenalg` are installed; otherwise it falls back to NetworkX Louvain.

The first run may download the spaCy model and sentence-transformer weights. The embedder loader also includes a local-cache fallback so later runs are more robust when network access is limited.

## Method Summary

### 1. Document Loading

Each `.txt` file is read as a document and assigned a deterministic UUID5 document identifier. The implementation currently hashes the relative filename and document text, which gives stable IDs for repeated indexing of the same file contents.

### 2. Sentence Segmentation and Entity Extraction

The pipeline loads `en_core_web_sm` with the lemmatizer disabled. It retains only the following named entity labels:

- `PERSON`
- `ORG`
- `PRODUCT`
- `GPE`
- `LOC`
- `EVENT`
- `FAC`
- `NORP`
- `WORK_OF_ART`

No separate relation extractor is used. Relational structure is induced downstream through shared hyperedge membership.

### 3. Alias Normalization

Entity mentions are canonicalized by:

- Unicode NFKC normalization,
- lowercasing,
- collapsing repeated whitespace and underscores,
- stripping leading/trailing punctuation and whitespace.

Canonical aliases are mapped to deterministic entity IDs via UUID5. Surface forms are preserved through an alias table, while the displayed entity title is chosen from the most frequent observed alias.

### 4. Hypergraph Construction

The index uses a document-grounded hypergraph with two edge types:

- one document hyperedge per document,
- one sentence hyperedge per sentence containing at least one retained entity.

For each hyperedge, the code stores incidence rows that link entity IDs to hyperedge IDs along with:

- entity role label,
- within-edge mention count,
- confidence,
- document ID,
- optional span ID.

In the current implementation, mention confidence is fixed at `1.0` because the chosen spaCy configuration does not expose calibrated mention-level confidence values.

### 5. Hyperedge Weighting

Hyperedge weight is computed from:

- edge type base weight,
- mean confidence,
- within-edge mention density,
- total mention count,
- edge size normalization.

Sentence hyperedges are upweighted relative to document hyperedges. This is intended to keep local evidence competitive with broader document context.

### 6. Pairwise Projection

For GraphRAG-style compatibility, the hypergraph is projected into a weighted 2-section graph. Each shared hyperedge contributes additive support to each unordered entity pair it contains, scaled by the hyperedge weight and normalized by hyperedge size.

The projected relation output records:

- source and target entity titles,
- relationship weight,
- combined degree,
- supporting text unit IDs,
- contributing hyperedge IDs.

### 7. Community Detection

Communities are detected over the projected graph.

- Preferred path: Leiden with `RBConfigurationVertexPartition` and `seed=42`.
- Fallback path: weighted Louvain via `networkx.community.louvain_communities` and `seed=42`.

The current package uses flat communities only. It does not implement hierarchical community refinement.

### 8. Community Reports

Community summaries are generated deterministically from:

- top entities by frequency,
- strongest within-community projected relationships,
- representative supporting spans.

These reports are not LLM-written in this cleaned package.

### 9. Embeddings

The retrieval layer uses two embedding channels.

Semantic channel:

- model: `all-MiniLM-L6-v2`
- entity text: `title + description`
- hyperedge text: document or span text
- truncation: first 1,024 characters
- normalization: L2

Structural channel:

- sparse entity-hyperedge incidence matrix
- entry value: `mention_count * confidence`
- decomposition: truncated SVD
- hyperedge structural vectors: incidence-weighted aggregation of entity structural vectors

Final vectors are the concatenation of semantic and structural embeddings followed by L2 normalization.

### 10. Query Path

At query time, the system:

1. loads stored hyperedge/entity vectors,
2. embeds the query,
3. scores hyperedges using semantic similarity plus local support terms,
4. aggregates entity and community relevance from selected hyperedges,
5. renders a response with evidence spans, documents, entities, and communities,
6. generates a lightweight short answer from retrieved evidence sentences.

The short answer in `cli.py` is heuristic and extractive. It is not meant to be a substitute for a fully generative answer model.

## Output Artifacts

A build writes the following core files into the chosen output directory.

- `documents.parquet`: one row per input document.
- `entities.parquet`: canonicalized entities with frequency, degree, and linked document/span IDs.
- `relationships.parquet`: projected pairwise relations.
- `projected_relationships.parquet`: richer relation table including shared hyperedge IDs.
- `hyperedges.parquet`: document and span hyperedges.
- `incidence.parquet`: entity-hyperedge membership table.
- `evidence_spans.parquet`: sentence-level evidence spans.
- `text_units.parquet`: GraphRAG-style text-unit table.
- `communities.parquet`: detected flat communities.
- `community_reports.parquet`: deterministic summaries for communities.
- `aliases.parquet`: alias-to-entity mapping.
- `entity_embeddings.parquet`: references into saved entity vector arrays.
- `hyperedge_embeddings.parquet`: references into saved hyperedge vector arrays.
- `embeddings/*.npy`: raw vector arrays.
- `graph.graphml`: projected graph snapshot.
- `stats.json`: summary counts.

## Reproducibility Notes

This package aims for stable, local, repeatable indexing behavior under a fixed software environment.

Reproducibility is strongest when:

- Python version is held constant,
- dependency versions are pinned or frozen,
- the same spaCy model version is used,
- the same sentence-transformer weights are used,
- the same input files and relative paths are used.

Potential sources of variation:

- dependency version drift,
- model version drift from external downloads,
- small differences in community detection behavior if optional graph backends differ,
- environmental differences in package builds.

For strict reproducibility, it is recommended to export a lockfile or environment spec after installation.

## What This Repo Is and Is Not

This repository is:

- a compact, runnable implementation of the method,
- suitable for demonstrations, ablations, and small-to-medium corpus experiments,
- usable as a base for method extensions.

This repository is not:

- the full archival research workspace,
- a benchmark suite,
- a production retrieval service,
- an LLM-based answer generator.

## Differences from the Larger Workspace

Compared with the original `cel-msft` workspace, this repo deliberately omits:

- notebooks,
- presentation files,
- comparison scripts,
- draft method and report documents,
- large precomputed outputs,
- topic-export utilities,
- dataset download/export helpers not required for the standalone package.

The goal here is repository clarity and GitHub publishability, not full workspace preservation.

## Extension Points

Natural places to extend the method include:

- replacing spaCy NER with a domain extractor,
- adding relation extraction before or after hypergraph construction,
- incorporating topic-aware or IDF-aware hyperedge weighting,
- replacing truncated SVD structural embeddings with bipartite walk-based methods,
- adding hierarchical community refinement,
- swapping the heuristic short answer with a controlled generative layer,
- adding evaluation scripts for retrieval quality and answer quality.
