# Notebook Development Guide for NLP Learning Repository

## 🎯 Purpose

This prompt guides AI assistants in developing, maintaining, and enhancing Jupyter notebooks for the NLP learning repository with a focus on beginner-friendly, Windows-first approaches.

## 📋 Core Requirements

### Notebook Structure

- **Self-contained execution**: Each notebook must run independently without dependencies on other notebook states
- **Beginner quick start**: Every notebook needs a clear "Beginner quick start" section at the top
- **Windows-first approach**: Provide Python fallback cells for all shell commands
- **Performance considerations**: Include warnings for long-running cells and offer quick demo options

### Essential Patterns

#### 1. Windows Fallback Pattern

```python
# Non-Windows (optional): shell-based download/extract
# !wget https://github.com/SalvatoreRa/tutorial/blob/main/datasets/IMDB.zip?raw=true
# !unzip IMDB.zip?raw=true

# Windows-friendly fallback (works everywhere)
def ensure_imdb_csv(csv_name='IMDB Dataset.csv', url='...'):
    if os.path.exists(csv_name):
        return csv_name
    # Implementation with urllib.request and zipfile
```

#### 2. Beginner Notes Pattern

```markdown
> Beginner notes: What to look for in the output
>
> - Run the next cell to see tokens, vocabulary size, and matrix shape
> - Look for: rows = documents, columns = vocabulary words
> - Try changing the input text and re-running to see differences
```

#### 3. NLTK Management Pattern

```python
# Ensure NLTK punkt is available (needed for word_tokenize)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# For newer NLTK versions, also ensure punkt_tab
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
```

#### 4. Performance-Conscious Training

```python
# Quick demo version (fast, for learning)
subset_size = 5000
df_subset = df.head(subset_size)

# Performance note: Full dataset training can take 10+ minutes
# For learning purposes, the subset provides the same concepts
```

#### 5. Cross-Platform DataFrame Styling

```python
try:
    styled = df.style.applymap(lambda v: 'background-color: #ffeeba' if v == 1 else '')
    display(styled)
except Exception:
    # Fallback if Styler/display not available
    df
```

## 🔧 Technical Standards

### Required Libraries (Pin These Versions)

- numpy==1.26.4
- pandas==2.3.1
- matplotlib==3.10.5
- seaborn==0.13.2
- scikit-learn==1.7.1
- nltk==3.9.1
- gensim==4.3.3
- torch==2.8.0
- umap-learn==0.5.9.post2
- tqdm==4.67.1
- wordcloud==1.9.4
- adjustText==1.3.0

### Code Quality Guidelines

- Use type hints where practical
- Keep functions small and focused
- Include docstrings for helper functions
- Prefer clear variable names over comments
- Use consistent naming patterns (e.g., `ensure_*` for download functions)

### Educational Best Practices

- Start with minimal examples before full implementations
- Show intermediate outputs (shapes, samples, visualizations)
- Explain what each visualization reveals
- Provide interactive elements where possible (user input, parameter tweaking)
- Include glossary references for technical terms

## 📊 Notebook-Specific Guidelines

### 01_text_representations.ipynb

- Focus on One-Hot, Bag of Words, TF/IDF
- Use small, clear examples (3-sentence corpus)
- Show matrix representations as DataFrames
- Include shape explanations and sparsity discussions

### 02_embeddings_visualization.ipynb

- Demonstrate Word2Vec training with subset options
- Include hierarchical clustering (dendrogram)
- Show t-SNE and UMAP visualizations
- Cap visualizations to reasonable word counts (2k max)

### 03_sentiment_classifier.ipynb

- Start with 1 epoch for pipeline validation
- Include GPU/CPU detection and fallbacks
- Show training progression with clear metrics
- Provide confusion matrix and accuracy analysis

### IntroductiontoAIAgents.ipynb

- Combine all concepts in logical progression
- Include comprehensive examples of each technique
- Provide end-to-end workflow demonstration
- Maintain beginner accessibility throughout

## 🚀 Development Workflow

### When Creating New Notebooks

1. Start with "Beginner quick start" section
2. Add necessary imports with version comments
3. Include Windows-friendly download functions
4. Add NLTK management cells
5. Implement core content with beginner notes
6. Include performance alternatives (quick vs. full)
7. Test all cells in clean environment
8. Add final summary and next steps

### When Updating Existing Notebooks

1. Preserve existing cell structure and IDs
2. Add Windows fallbacks where missing
3. Enhance beginner notes for clarity
4. Update library versions in requirements.txt
5. Test cross-platform compatibility
6. Verify performance with both quick and full options

## 🎓 Educational Philosophy

### For Beginners

- Prefer clarity over brevity
- Show intermediate steps explicitly
- Explain what outputs mean
- Provide safe experimentation guidelines
- Link to additional resources

### For Windows Users

- Always provide Python alternatives to shell commands
- Test file path handling
- Include PowerShell examples in documentation
- Ensure compatibility with common Windows Python setups

### For Performance

- Default to quick demos for interactivity
- Clearly mark long-running operations
- Provide subset options for large datasets
- Include progress bars (tqdm) for longer operations

## ⚠️ Common Pitfalls to Avoid

1. **Shell Dependencies**: Never rely only on !wget/!unzip commands
2. **Cross-Notebook State**: Don't assume variables from other notebooks exist
3. **Large Downloads**: Always check if files exist before downloading
4. **NLTK Assumptions**: Always verify tokenizer availability
5. **Display Dependencies**: Use try/except for notebook-specific display features
6. **Hardcoded Paths**: Use os.path.join for cross-platform compatibility
7. **Memory Issues**: Provide subset options for memory-constrained environments

## 📝 Documentation Standards

### Cell Comments

- Explain non-obvious operations
- Include performance expectations
- Reference external resources when helpful
- Provide parameter tuning guidance

### Markdown Cells

- Use clear headings and subsections
- Include bullet points for key concepts
- Add visual breaks between major sections
- Provide "What's next" guidance between sections

### Error Handling

- Graceful degradation for optional features
- Clear error messages for common issues
- Fallback options for failed operations
- Recovery suggestions for blocked installations
