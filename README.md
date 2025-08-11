# Building AI Agents with LLMs, RAG, and Knowledge Graphs

A practical learning repo for building modern AI agents with LLMs. Includes notebooks and small apps covering transformers, RAG pipelines, knowledge graphs, reinforcement learning, and a multi-agent Streamlit travel planner.

## Learning Source

This repository is inspired by and follows concepts from the book:

**Building AI Agents with LLMs, RAG, and Knowledge Graphs** (Packt Publishing)

For more details, visit [Packt Publishing](https://www.packtpub.com/en-us/product/building-ai-agents-with-llms-rag-and-knowledge-graphs-9781835080382).

## 🛠️ Prerequisites

- Windows 11 + PowerShell
- Python 3.12.x (recommended for smooth SciPy/gensim installs on Windows)
 	- If you’re on Python 3.13, some scientific packages may try to build from source. Prefer 3.12 or use Conda.
- OpenAI API key (only for later chapters/apps)
- Basic programming knowledge and curiosity

## 🚀 Setup & Quick Start (Windows + PowerShell)

```powershell
# Clone the repository
git clone https://github.com/Swamy-s-Tech-Skills-Academy-AI-ML-Data/llm-agents-learning.git
cd llm-agents-learning

# Create a Python 3.12 virtual environment (recommended)
py -3.12 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install required Python packages
pip install numpy pandas matplotlib scikit-learn seaborn nltk

# Optional: install extras when you reach embeddings/visualizations/DL
# (gensim pulls in scipy; Python 3.12 avoids source builds on Windows)
pip install gensim tqdm adjustText umap-learn torch wordcloud

# Save your environment
pip freeze > requirements.txt

# Install dependencies from requirements.txt (if any)
pip install -r requirements.txt
```

## 📦 Required Python Packages

See [docs/Chapter1.md](docs/Chapter1.md) for a full list of required packages and their purpose for Chapter 1.

> **Note:** If you are starting fresh, `requirements.txt` will be empty. You can manually install packages as you progress and update `requirements.txt` accordingly.

## ✅ Verify your Python setup

```powershell
# Print the active interpreter version (should show 3.12.x)
python -c "import sys; print(sys.version)"

# List installed Python interpreters (Windows launcher)
py -0p
```

## 📒 Start with the first notebook

- Open `src/ch1/IntroductiontoAIAgents.ipynb` in VS Code
- Select the `.venv` interpreter as the Jupyter kernel
- Run cells in order; the first sections (One‑Hot, BoW, TF/IDF) only need `numpy`
