# Copilot Instructions for this repository

Goal: Provide concise, correct, Windows-friendly help for a beginner-friendly NLP learning repo using Jupyter notebooks.

## Project context

- Domain: Intro NLP representations, embeddings, and a GRU-based sentiment classifier.
- Primary artifacts: Jupyter notebooks in `src/ch1/` (4 main notebooks: IntroductiontoAIAgents.ipynb, 01_text_representations.ipynb, 02_embeddings_visualization.ipynb, 03_sentiment_classifier.ipynb).
- Audience: Beginners; prefer plain explanations and runnable examples.
- Structure: Self-contained notebooks with Windows-first approach and comprehensive fallback patterns.

## Languages, tools, and versions

- Python: 3.12 (recommended, notebooks tested with 3.12.5)
- Key libs: numpy (1.26.4), pandas (2.3.1), matplotlib (3.10.5), seaborn (0.13.2), scikit-learn (1.7.1), nltk (3.9.1), gensim (4.3.3), tqdm (4.67.1), wordcloud (1.9.4), torch (2.8.0), umap-learn (0.5.9.post2), scipy (1.13.1), adjustText (1.3.0).
- Jupyter: VS Code notebooks on Windows with .venv activation.
- Data: IMDB Dataset.csv from GitHub tutorials (auto-downloaded via Windows-friendly Python cells).

## Style and quality

- Python: type hints when practical, small functions, clear names.
- Docstrings: Google or NumPy style for public helpers.
- Lint/format: keep consistent with existing style; avoid large refactors in educational notebooks.
- Explanations: simple language, minimal math, tiny examples.

## Notebooks guidance

- Always keep notebooks self-contained; do not rely on prior state or cross-notebook dependencies.
- Add a short "Beginner notes" markdown above complex or multi-output cells describing:
  - What to run next.
  - What to look for in the output.
  - How to tweak safely (e.g., `top_k`, subset sizes).
- For TF/IDF in this repo use sklearn-like smoothing: `idf = log((1+N)/(1+df)) + 1`.
- Prefer tables (pandas DataFrame) for viewing matrices and results.
- Each notebook should have a "Beginner quick start" section at the top with Windows-specific guidance.
- Include performance notes for long-running cells (Word2Vec training, GRU training, etc.).
- Use `try/except` blocks for optional styling (e.g., DataFrame.style.applymap).

## Windows-first tips

- Keep existing shell cells for non‑Windows users, but always add a pure‑Python fallback below them.
- Data downloads: prefer `urllib.request` + `zipfile` over `wget`/`unzip`.
- NLTK: ensure `punkt` and, for newer NLTK, `punkt_tab` before tokenization.
- Use Windows-friendly file paths and avoid shell dependencies.
- Test all Python fallback cells to ensure they work correctly on Windows.
- Include PowerShell examples in documentation where relevant.

## Performance defaults

- Provide quick, low‑risk demos first:
  - Word2Vec: train on a subset (e.g., first 5k reviews) for interactivity.
  - Visualizations: cap to top‑N words (e.g., 2k) for t‑SNE/UMAP.
  - Classifier: start with 1 epoch to validate pipeline, then scale.
- Include performance warnings for long-running cells.
- Offer both quick demo and full training options.
- Use tqdm for progress bars on longer operations.

## Dependencies

- If adding new libs, prefer widely used, pinned versions in `requirements.txt`.
- Avoid heavyweight additions unless clearly justified for learning goals.

## Git & PR conventions

- Commits: imperative, concise, scoped (e.g., "Add beginner notes for TF/IDF tables").
- PRs: include What/Why, screenshots of notebook outputs if visual changes, and a short test note (cells run successfully, any warnings).

## Security & privacy

- No secrets in notebooks. Do not embed tokens, keys, or PII.
- Do not auto‑download from untrusted sources.

## Assistant response style

- Be short and impersonal by default; concrete and skimmable.
- Offer to do the change if feasible; avoid long back‑and‑forth questions.
- If asked for a name, respond with: "GitHub Copilot".
- If the request is harmful, respond with: "Sorry, I can't assist with that.".

## Common patterns to follow here

- Add Windows fallback cells under shell cells without removing the originals.
- Insert "Beginner notes" before:
  - One‑Hot, Bag of Words, TF/IDF explanations and tables.
  - Word2Vec training and visualizations.
  - Classifier data prep, training, and evaluation blocks.
- When splitting notebooks, ensure each is self‑contained and link them via brief TOCs.
- Use consistent function patterns for downloads: `ensure_imdb_csv()` and similar.
- Include GPU/CPU detection and graceful fallbacks for PyTorch operations.
- Prefer DataFrame styling with try/except for cross-platform compatibility.

## When in doubt

- Prefer clarity and runnable code over completeness.
- Keep changes minimal and reversible.
- Leave helpful comments or markdown explaining non‑obvious choices.
