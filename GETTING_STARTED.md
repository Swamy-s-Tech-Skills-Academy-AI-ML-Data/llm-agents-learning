# Getting Started (Windows, PowerShell)

This guide helps you run the chapter projects locally and publish your own GitHub repository.

## Quick start (TL;DR)

```powershell
# Chapter 9 (CLI)
cd d:\PacktPub\Modern-AI-Agents\chr9
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\Multi_Model–Travel_Planning_System_v_0_5.py

# Chapter 10 (Streamlit)
cd d:\PacktPub\Modern-AI-Agents\chr10
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir .streamlit 2>$null
ni .streamlit\secrets.toml -ItemType File -Force | Out-Null
# Edit secrets.toml with your OpenAI key under [general]
streamlit run .\Multi_Model–Travel_Planning_System_streamlit_v_0_2.py
```

## Prerequisites

- Windows 10/11
- Python 3.10 or 3.11 recommended
- PowerShell in VS Code (default shell)
- Git installed

## Run Chapter 9 (CLI multi‑agent travel planner)

```powershell
# From repo root
cd d:\PacktPub\Modern-AI-Agents\chr9
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\Multi_Model–Travel_Planning_System_v_0_5.py
```

Notes

- First run downloads Hugging Face models (can take a few minutes).

## Run Chapter 10 (Streamlit app)

1. Create environment and install

```powershell
cd d:\PacktPub\Modern-AI-Agents\chr10
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

1. Add OpenAI key in Streamlit secrets

```powershell
# Create folder then file
mkdir .streamlit 2>$null
ni .streamlit\secrets.toml -ItemType File -Force | Out-Null
```

Edit `.streamlit/secrets.toml` and add:

```toml
[general]
openai_api_key = "sk-..."
```

1. Start the app

```powershell
streamlit run .\Multi_Model–Travel_Planning_System_streamlit_v_0_2.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Run notebooks

- Activate the matching venv first (chrX/.venv).
- Open the `.ipynb` in VS Code and select that interpreter as the kernel.

## GPU requirements

- Not required for the book projects.
- Chapter 9 and 10 examples run well on CPU. The Streamlit app uses OpenAI API for GPT‑4 (compute is remote).
- The RL Super Mario example (chr8) trains much faster on a GPU, but it can run on CPU (slower).

Optional CPU-only tips:

```powershell
# Install CPU-only PyTorch (if your default install struggles)
pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch

# Force CPU at runtime for this session
$env:CUDA_VISIBLE_DEVICES = "-1"
```

In code, you can also explicitly pin devices:

```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer

pipe = pipeline("text-generation", model="gpt2", device_map="cpu")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
```

## Create your own GitHub repository

1. Create an empty repo on GitHub (no README/.gitignore).

2. Initialize and push from local root:

```powershell
cd d:\PacktPub\Modern-AI-Agents
git init
git branch -M main
git add .
git commit -m "Initial commit from book workspace"
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If you previously had a different origin:

```powershell
git remote remove origin
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## Recommended .gitignore (optional)

Add a `.gitignore` at the repo root:

```gitignore
# Python
.venv/
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Streamlit secrets
.streamlit/secrets.toml

# Caches / models
.cache/
.hf_cache/
```

## Troubleshooting

- torch install issues: ensure Python 3.10/3.11; allow pip to resolve a compatible wheel.
- Large first‑run downloads (transformers, sentence‑transformers) are normal.
- Tokenizers parallelism warnings are disabled in the Streamlit app; if needed elsewhere, set `TOKENIZERS_PARALLELISM=false`.
- Keep your OpenAI key only in `.streamlit/secrets.toml` (don’t commit it).
- Activation blocked on Windows: if `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell, then activate again.
- Filename dash gotcha: the travel planner filenames use an en dash (–). Copy/paste the name from Explorer or this file to avoid a regular hyphen (-) mismatch.
- Streamlit secrets location: `.streamlit/secrets.toml` must live inside the `chr10` folder (same folder as the Streamlit script).
- Deactivate the venv: run `deactivate` in the same PowerShell session when you’re done.
