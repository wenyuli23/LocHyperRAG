"""Command-line interface for LoCHyperRAG."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import spacy
import typer

from .core import build_hypergraph_index, query_hypergraph_index

app = typer.Typer(add_completion=False, help="Local, deterministic HyperGraphRAG indexing and retrieval.")

_SIMPLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "up",
    "use",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


def normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def normalized_query_terms(text: str) -> set[str]:
    return {
        normalize_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in _SIMPLE_STOPWORDS
    }


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if part.strip()]


def sentence_score(sentence: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    sentence_terms = {
        normalize_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", sentence.lower())
        if len(token) > 2 and token not in _SIMPLE_STOPWORDS
    }
    if not sentence_terms:
        return 0.0
    return len(sentence_terms & query_terms) / max(len(query_terms), 1)


def short_answer_from_result(query: str, result: dict) -> str:
    query_terms = normalized_query_terms(query)
    candidates = []
    for row in result.get("local_hyperedges", []):
        text = re.sub(r"(?im)^subject:\s*[^\r\n]+", "", str(row.get("text") or "")).strip()
        retrieval_score = float(row.get("retrieval_score", 0.0))
        source = str(row.get("type") or "")
        for sentence in split_sentences(text):
            if len(sentence.split()) < 4:
                continue
            score = 0.65 * retrieval_score + 0.35 * sentence_score(sentence, query_terms)
            sentence_lower = sentence.lower()
            if source == "span":
                score += 0.05
            if any(marker in sentence_lower for marker in ["support replied", "integrates", "available", "planned", "not available", "explained"]):
                score += 0.08
            if any(marker in sentence_lower for marker in [" asked ", "question", "wanted guidance", "wants one system"]):
                score -= 0.08
            candidates.append((score, sentence))

    if not candidates:
        return "I could not compose a short answer from the retrieved evidence."

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence in candidates:
        if sentence in seen:
            continue
        selected.append(sentence)
        seen.add(sentence)
        if len(selected) == 2:
            break
    return " ".join(selected)


def formatted_response(query: str, result: dict) -> str:
    answer = short_answer_from_result(query, result)
    return f"# Short answer\n\n{answer}\n\n{result['response']}"


@app.command()
def doctor() -> None:
    """Check whether the local environment is ready to run the demo."""
    packages = [
        "networkx",
        "numpy",
        "pandas",
        "pyarrow",
        "sklearn",
        "scipy",
        "sentence_transformers",
        "spacy",
        "typer",
        "yaml",
    ]
    missing = [package for package in packages if importlib.util.find_spec(package) is None]

    typer.echo("LoCHyperRAG environment check")
    typer.echo("-----------------------------")
    if missing:
        typer.echo("Missing Python packages:")
        for package in missing:
            typer.echo(f"- {package}")
    else:
        typer.echo("All required Python packages are installed.")

    if spacy.util.is_package("en_core_web_sm"):
        typer.echo("spaCy model 'en_core_web_sm' is installed.")
    else:
        typer.echo("spaCy model 'en_core_web_sm' is missing.")
        typer.echo("Install it with: python -m spacy download en_core_web_sm")

    if not missing and spacy.util.is_package("en_core_web_sm"):
        typer.echo("Status: ready to run.")
    else:
        typer.echo("Status: finish the missing setup steps above, then run this command again.")


@app.command("build")
def build_command(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="Folder that contains .txt files."),
    output_dir: Path = typer.Option(..., file_okay=False, dir_okay=True, help="Folder where the index files will be written."),
    max_docs: int | None = typer.Option(None, help="Optional limit for a small test run."),
) -> None:
    """Build an index from a folder of plain-text files."""
    stats = build_hypergraph_index(root=Path.cwd(), input_dir=input_dir, output_dir=output_dir, max_docs=max_docs)
    typer.echo(json.dumps(stats, indent=2))


@app.command("ask")
def ask_command(
    output_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, help="Folder that already contains a built index."),
    query: str = typer.Option(..., help="Question to ask the index."),
    top_k_hyperedges: int = typer.Option(8, help="How many top hyperedges to retrieve."),
    top_k_entities: int = typer.Option(10, help="How many top entities to show."),
    top_k_communities: int = typer.Option(3, help="How many top communities to show."),
    as_json: bool = typer.Option(False, help="Print raw JSON instead of the formatted answer."),
) -> None:
    """Ask a question against an existing index."""
    result = query_hypergraph_index(
        root=Path.cwd(),
        data_dir=output_dir,
        query=query,
        top_k_hyperedges=top_k_hyperedges,
        top_k_entities=top_k_entities,
        top_k_communities=top_k_communities,
    )
    if as_json:
        result["answer"] = short_answer_from_result(query, result)
        typer.echo(json.dumps(result, indent=2))
        return
    typer.echo(formatted_response(query, result))


@app.command()
def demo(
    query: str = typer.Option(
        "Which ecosystems does the current HomeLink Hub integrate with?",
        help="Demo question to ask after the sample index is built.",
    ),
    output_dir: Path | None = typer.Option(None, file_okay=False, dir_okay=True, help="Optional custom output folder for the demo index."),
) -> None:
    """Build the bundled demo index and ask one sample question."""
    repo_root = Path(__file__).resolve().parents[2]
    demo_input_dir = repo_root / "demo" / "input"
    demo_output_dir = output_dir or (repo_root / "demo" / "output")

    typer.echo("Building the demo index...")
    stats = build_hypergraph_index(root=repo_root, input_dir=demo_input_dir, output_dir=demo_output_dir)
    typer.echo(json.dumps(stats, indent=2))
    typer.echo("")
    typer.echo("Running the demo question...")
    typer.echo("")
    result = query_hypergraph_index(root=repo_root, data_dir=demo_output_dir, query=query)
    typer.echo(formatted_response(query, result))


if __name__ == "__main__":
    app()
