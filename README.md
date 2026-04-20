# LoCHyperRAG

LoCHyperRAG is a small, local, beginner-friendly version of the HyperGraphRAG method from the research workspace. It is designed to be publishable on GitHub and runnable by a student on a normal laptop without setting up an API key or the full experiment environment.

This cleaned repo keeps only the parts needed to do one real thing:

1. read a folder of plain-text documents,
2. build a document-and-sentence hypergraph index,
3. ask a question,
4. get back a short answer plus supporting evidence.

It does **not** include the work-in-progress material from the original workspace such as notebooks, presentations, comparison experiments, large generated outputs, or draft reports.

## What this repository contains

- `src/lochyperrag/core.py`: the indexing and retrieval method copied from the working implementation.
- `src/lochyperrag/cli.py`: the beginner-friendly command-line interface.
- `demo/input/`: five tiny example text files.
- `requirements.txt` and `pyproject.toml`: install files.
- this `README.md`: a step-by-step guide aimed at non-CS students.

## What you need before you start

You need:

- a laptop with Python 3.10, 3.11, or 3.12 installed,
- internet on the first run so Python can install packages and download the small language/model files,
- a terminal window.

If you are on Windows, use **PowerShell** or **Windows Terminal**.
If you are on macOS or Linux, use the regular terminal app.

## Very short version

If you already know how to use Python, the fastest path is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m spacy download en_core_web_sm
lochyperrag doctor
lochyperrag demo
```

If the last command works, your setup is correct.

## Full beginner session

This section assumes you have never run a Python project before.

### Step 1: open the project folder in a terminal

After cloning or downloading this repository, open a terminal in the repository folder.

On Windows, a normal session looks like this:

```powershell
cd path\to\lochyperrag-github
```

To check that you are in the right place, run:

```powershell
Get-ChildItem
```

You should see files such as `README.md`, `pyproject.toml`, and a folder named `src`.

### Step 2: make a private Python environment

A virtual environment keeps this project separate from other Python projects on your computer.

Windows PowerShell:

```powershell
python -m venv .venv
```

macOS/Linux:

```bash
python3 -m venv .venv
```

If you get an error like `python is not recognized`, install Python first from https://www.python.org/downloads/ and then reopen the terminal.

### Step 3: turn the environment on

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the command, run this once in the same terminal window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then run the activate command again.

macOS/Linux:

```bash
source .venv/bin/activate
```

When the environment is active, the terminal usually shows `(.venv)` at the beginning of the line.

### Step 4: upgrade pip

```powershell
python -m pip install --upgrade pip
```

This makes package installation smoother.

### Step 5: install the project

Install the package in editable mode:

```powershell
pip install -e .
```

What this does:

- installs the required Python libraries,
- makes the `lochyperrag` command available in this environment,
- links the current folder so local code changes are picked up immediately.

### Step 6: install the spaCy English model

LoCHyperRAG uses spaCy for sentence splitting and named entity recognition.

Run:

```powershell
python -m spacy download en_core_web_sm
```

This is a one-time setup step.

### Step 7: check that everything is ready

Run:

```powershell
lochyperrag doctor
```

If everything is set up correctly, you should see messages telling you that:

- the Python packages are installed,
- the `en_core_web_sm` model is installed,
- the status is ready.

If the `lochyperrag` command is not found, use this equivalent command instead:

```powershell
python -m lochyperrag.cli doctor
```

## Run the included demo

This is the easiest way to prove the method works before you use your own data.

### Step 8: build the demo index and ask the demo question

Run:

```powershell
lochyperrag demo
```

If the command-line shortcut is not available, use:

```powershell
python -m lochyperrag.cli demo
```

This single command will:

1. read the text files in `demo/input/`,
2. build a hypergraph index in `demo/output/`,
3. ask a sample question,
4. print a short answer and supporting evidence.

### Step 9: what answer should you expect?

Your exact scores may differ slightly, but the answer should say something close to this:

> The current HomeLink Hub integrates with Amazon Alexa and Google Assistant. Apple HomeKit is planned for a later release and is not available in the current version.

If you see that idea in the output, the demo is working.

### Step 10: ask your own question about the demo data

Once the demo index exists, you can ask new questions without rebuilding it every time.

Example:

```powershell
lochyperrag ask --output-dir demo/output --query "Which ecosystems are available now?"
```

Another example:

```powershell
lochyperrag ask --output-dir demo/output --query "What billing problem was reported?"
```

The output will contain:

- a short answer,
- the most relevant evidence spans,
- the most relevant supporting documents,
- top entities,
- top communities.
## Use your own `.txt` files

After the demo works, you can index your own folder of plain-text documents.

### Step 1: make a folder for your documents

For example:

```powershell
New-Item -ItemType Directory my_documents
```

Add plain-text files ending in `.txt` to that folder.

Important:

- one file = one document,
- `.txt` files work directly,
- PDFs, Word files, and spreadsheets should be converted to `.txt` first.

### Step 2: build an index for your own folder

```powershell
lochyperrag build --input-dir my_documents --output-dir my_output
```

What this command does:

- reads all `.txt` files under `my_documents`,
- extracts named entities,
- creates document hyperedges and sentence hyperedges,
- projects the hypergraph into pairwise relationships,
- finds communities,
- creates semantic and structural embeddings,
- writes the results to `my_output`.

### Step 3: ask a question

```powershell
lochyperrag ask --output-dir my_output --query "What are the main problems customers reported?"
```

### Step 4: ask more questions

You do **not** need to rebuild the index for every question.
You only rebuild when the input documents change.

## Understanding the output folder

After you run `build` or `demo`, the output folder will contain files such as:

- `documents.parquet`: one row per source document,
- `entities.parquet`: the entities detected by spaCy,
- `hyperedges.parquet`: document-level and sentence-level hyperedges,
- `incidence.parquet`: which entities belong to which hyperedges,
- `relationships.parquet`: projected pairwise relationships,
- `communities.parquet`: detected communities,
- `community_reports.parquet`: deterministic community summaries,
- `graph.graphml`: a graph file you can inspect later,
- `embeddings/`: saved semantic and structural vector files,
- `stats.json`: a quick summary of how many documents, entities, and relationships were created.

You do not need to open these files to use the system, but they are there if you want to inspect what happened.

## What the method is doing in plain English

LoCHyperRAG does not treat knowledge as only simple pairs like `A -> B`.
Instead, it keeps group evidence.

In plain language, the method does this:

1. It reads each text document.
2. It finds named entities such as people, organizations, products, and places.
3. It makes one hyperedge for the whole document and one hyperedge for each sentence with entities.
4. It weights those hyperedges so short, dense evidence can matter more.
5. It converts the hypergraph to a weighted graph so communities can be detected.
6. It builds semantic + structural embeddings.
7. At question time, it retrieves relevant hyperedges first, then expands to entities and communities.
8. It returns a short answer based on the best evidence spans.

## Common problems and fixes

### Problem: `python` is not recognized

Fix:

- install Python from https://www.python.org/downloads/
- reopen the terminal,
- try again.

### Problem: `lochyperrag` is not recognized

Fix:

- make sure the virtual environment is activated,
- make sure you ran `pip install -e .`,
- or use `python -m lochyperrag.cli ...` instead.

### Problem: spaCy model missing

Fix:

```powershell
python -m spacy download en_core_web_sm
```

### Problem: first run takes a while

This is normal.
On the first run, `sentence-transformers` may download the embedding model `all-MiniLM-L6-v2`.
Later runs are faster because the model is cached.

### Problem: my answer is weak or incomplete

Try one or more of these:

- ask a more specific question,
- make sure your documents are plain, readable text,
- split very large files into smaller topic-focused files,
- check that the important names or products actually appear in the text.

### Problem: I changed my input files, but the results look old

Fix:

- delete the old output folder,
- rebuild the index,
- then ask the question again.

## Optional: better community detection

By default, the code will use NetworkX Louvain community detection.
If you want Leiden instead, install the optional packages:

```powershell
pip install igraph leidenalg
```

You do **not** need these packages for the demo.

## Recommended first commands to copy-paste

If you want the safest copy-paste order, use this exact sequence in a fresh terminal:

```powershell
cd path\to\lochyperrag-github
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m spacy download en_core_web_sm
lochyperrag doctor
lochyperrag demo
lochyperrag ask --output-dir demo/output --query "Which ecosystems are available now?"
```

## Repository layout

```text
lochyperrag-github/
├── README.md
├── pyproject.toml
├── requirements.txt
├── demo/
│   └── input/
│       ├── 01_smart_home_customer_question.txt
│       ├── 02_smart_home_support_reply.txt
│       ├── 03_billing_renewal_problem.txt
│       ├── 04_medical_data_sync_issue.txt
│       └── 05_project_management_integrations.txt
└── src/
    └── lochyperrag/
        ├── __init__.py
        ├── cli.py
        └── core.py
```

## Final note

This repository is intentionally small and clean.
It is meant to be a publishable method repo, not a full research archive.
If you can run the demo and get the smart-home answer, you are ready to try your own documents.
