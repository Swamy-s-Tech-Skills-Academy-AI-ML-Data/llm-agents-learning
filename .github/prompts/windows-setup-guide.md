# Windows Setup and Environment Guide

## 🎯 Purpose

This prompt helps AI assistants guide users through setting up and maintaining a Windows-compatible Python environment for the NLP learning repository.

## 🔧 Environment Setup

### Prerequisites

- Windows 10/11
- Python 3.12.5 (recommended)
- PowerShell (default in VS Code)
- Git for Windows
- VS Code with Python and Jupyter extensions

### Virtual Environment Creation

```powershell
# Navigate to project root
cd d:\STSAAIMLDT\llm-agents-learning

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### VS Code Configuration

#### Jupyter Kernel Setup

1. Open VS Code in project directory
2. Open any .ipynb file
3. Click "Select Kernel" in top-right
4. Choose "Python Environments..."
5. Select the .venv interpreter

#### PowerShell Terminal

1. Open VS Code terminal (Ctrl + `)
2. Ensure PowerShell is selected
3. Activate .venv if not auto-activated
4. Verify Python version: `python --version`

## 📦 Dependency Management

### Requirements.txt Management

The repository uses pinned versions for reproducibility:

- numpy==1.26.4 (Core numerical computing)
- pandas==2.3.1 (Data manipulation)
- matplotlib==3.10.5 (Basic plotting)
- seaborn==0.13.2 (Statistical visualization)
- scikit-learn==1.7.1 (Machine learning)
- nltk==3.9.1 (Natural language toolkit)
- gensim==4.3.3 (Word embeddings)
- torch==2.8.0 (Deep learning)
- tqdm==4.67.1 (Progress bars)
- wordcloud==1.9.4 (Word cloud generation)
- umap-learn==0.5.9.post2 (Dimensionality reduction)
- adjustText==1.3.0 (Text positioning in plots)

### Adding New Dependencies

```powershell
# Install new package
pip install package-name==version

# Update requirements.txt
pip freeze > requirements.txt

# Or manually add to requirements.txt with specific version
```

## 🛠️ Common Windows Issues and Solutions

### NLTK Data Download Issues

```python
# Manual NLTK data download for Windows
import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### File Path Handling

```python
import os
from pathlib import Path

# Windows-safe path construction
data_dir = Path.cwd() / "data"
data_dir.mkdir(exist_ok=True)

# Cross-platform file operations
csv_path = data_dir / "IMDB Dataset.csv"
```

### Memory Management

```python
# Monitor memory usage on Windows
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
```

## 🔍 Troubleshooting Guide

### Environment Issues

#### Problem: Virtual environment not activating

```powershell
# Check execution policy
Get-ExecutionPolicy

# If restricted, temporarily allow scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate environment
.\.venv\Scripts\Activate.ps1
```

#### Problem: Package installation fails

```powershell
# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Install specific package with verbose output
pip install -v package-name
```

### Jupyter Issues

#### Problem: Kernel not found

1. Ensure .venv is activated
2. Install ipykernel: `pip install ipykernel`
3. Register kernel: `python -m ipykernel install --user --name=.venv`
4. Restart VS Code
5. Select the correct kernel

#### Problem: Import errors in notebooks

```python
# Verify current environment
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Check if package is installed
try:
    import package_name
    print(f"Package version: {package_name.__version__}")
except ImportError as e:
    print(f"Import error: {e}")
```

## 📋 Environment Verification Checklist

### Quick Health Check

```powershell
# 1. Check Python version
python --version

# 2. Check virtual environment
where python

# 3. List installed packages
pip list

# 4. Verify key packages
python -c "import numpy, pandas, matplotlib, torch; print('All key packages imported successfully')"

# 5. Check NLTK data
python -c "import nltk; nltk.data.find('tokenizers/punkt'); print('NLTK punkt available')"
```

### Performance Verification

```python
# Check if GPU is available (for PyTorch)
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Test basic operations
import numpy as np
import pandas as pd

# Create test data
test_array = np.random.random((1000, 100))
test_df = pd.DataFrame(test_array)
print(f"Test array shape: {test_array.shape}")
print(f"Test DataFrame shape: {test_df.shape}")
```

## 🔄 Maintenance Tasks

### Regular Updates

```powershell
# Weekly: Update pip
python -m pip install --upgrade pip

# Monthly: Check for package updates
pip list --outdated

# Update specific packages (test in separate environment first)
pip install --upgrade package-name
```

### Backup and Recovery

```powershell
# Backup current environment
pip freeze > requirements-backup-$(Get-Date -Format "yyyy-MM-dd").txt

# Recreate environment from backup
pip install -r requirements-backup-2024-01-15.txt
```

### Clean Installation

```powershell
# Remove virtual environment
Remove-Item .venv -Recurse -Force

# Recreate from scratch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🎯 Best Practices for Windows Development

### File Management

- Use absolute paths when possible
- Prefer Path objects over string concatenation
- Check file existence before operations
- Handle Windows-specific file locking issues

### Resource Management

- Monitor memory usage with large datasets
- Use generators for large file processing
- Implement proper cleanup in finally blocks
- Consider Windows antivirus impact on file operations

### Development Workflow

- Commit requirements.txt changes
- Test in clean virtual environment before sharing
- Document Windows-specific setup steps
- Provide fallback solutions for common issues
