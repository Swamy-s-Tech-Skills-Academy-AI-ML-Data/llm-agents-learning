# Copilot Instructions for this repository

Goal: Provide concise, correct, Windows-friendly help for a beginner-friendly NLP learning repo using Jupyter notebooks.

## Project context
- Domain: Intro NLP representations, embeddings, and a GRU-based sentiment classifier.
- Primary artifacts: Jupyter notebooks in `src/ch1/`.
- Audience: Beginners; prefer plain explanations and runnable examples.

## Languages, tools, and versions
- Python: 3.12 (recommended)
- Key libs: numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, gensim, tqdm, wordcloud, torch, umap-learn, scipy.
- Jupyter: VS Code notebooks on Windows.

## Style and quality
- Python: type hints when practical, small functions, clear names.
- Docstrings: Google or NumPy style for public helpers.
- Lint/format: keep consistent with existing style; avoid large refactors in educational notebooks.
- Explanations: simple language, minimal math, tiny examples.

## Notebooks guidance
- Always keep notebooks self-contained; do not rely on prior state.
- Add a short “Beginner notes” markdown above complex or multi-output cells describing:
  - What to run next.
  - What to look for in the output.
  - How to tweak safely (e.g., `top_k`, subset sizes).
- For TF/IDF in this repo use sklearn-like smoothing: `idf = log((1+N)/(1+df)) + 1`.
- Prefer tables (pandas DataFrame) for viewing matrices.

## Windows-first tips
- Keep existing shell cells for non‑Windows users, but always add a pure‑Python fallback below them.
- Data downloads: prefer `urllib.request` + `zipfile` over `wget`/`unzip`.
- NLTK: ensure `punkt` and, for newer NLTK, `punkt_tab` before tokenization.

## Performance defaults
- Provide quick, low‑risk demos first:
  - Word2Vec: train on a subset (e.g., first 5k reviews) for interactivity.
  - Visualizations: cap to top‑N words (e.g., 2k) for t‑SNE/UMAP.
  - Classifier: start with 1 epoch to validate pipeline, then scale.

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
- Insert “Beginner notes” before:
  - One‑Hot, Bag of Words, TF/IDF explanations and tables.
  - Word2Vec training and visualizations.
  - Classifier data prep, training, and evaluation blocks.
- When splitting notebooks, ensure each is self‑contained and link them via brief TOCs.

## When in doubt
- Prefer clarity and runnable code over completeness.
- Keep changes minimal and reversible.
- Leave helpful comments or markdown explaining non‑obvious choices.
