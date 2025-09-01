# Data Science Workflow and Best Practices

## 🎯 Purpose

This prompt guides AI assistants in implementing robust data science workflows for the NLP learning repository, emphasizing reproducibility, error handling, and educational value.

## 📊 Data Pipeline Standards

### Data Acquisition

#### Reliable Download Pattern

```python
def ensure_dataset(filename, url, backup_urls=None):
    """
    Download dataset with fallback options and verification.
    
    Args:
        filename: Local filename to save
        url: Primary download URL
        backup_urls: List of alternative URLs
    
    Returns:
        Path to downloaded file
    """
    if os.path.exists(filename):
        print(f"✅ {filename} already exists")
        return filename
    
    urls_to_try = [url] + (backup_urls or [])
    
    for attempt_url in urls_to_try:
        try:
            print(f"📥 Downloading from {attempt_url}")
            urllib.request.urlretrieve(attempt_url, filename + '.tmp')
            
            # Verify download
            if os.path.getsize(filename + '.tmp') > 0:
                os.rename(filename + '.tmp', filename)
                print(f"✅ Successfully downloaded {filename}")
                return filename
                
        except Exception as e:
            print(f"❌ Failed to download from {attempt_url}: {e}")
            continue
    
    raise FileNotFoundError(f"Could not download {filename} from any source")
```

#### Data Validation

```python
def validate_dataset(df, expected_columns, min_rows=1000):
    """Validate dataset meets expectations."""
    assert isinstance(df, pd.DataFrame), "Data must be a pandas DataFrame"
    assert len(df) >= min_rows, f"Dataset too small: {len(df)} < {min_rows}"
    
    missing_cols = set(expected_columns) - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"
    
    print(f"✅ Dataset validation passed: {len(df):,} rows, {len(df.columns)} columns")
    return True
```

### Data Preprocessing

#### Text Cleaning Pipeline

```python
def preprocess_text(text, 
                   lowercase=True, 
                   remove_html=True, 
                   remove_special_chars=True,
                   min_length=3):
    """
    Standardized text preprocessing pipeline.
    
    Educational note: Each step serves a specific purpose in NLP.
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text)
    if lowercase:
        text = text.lower()
    
    # Remove HTML tags
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters but keep spaces
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Filter by length
    if len(text) < min_length:
        return ""
    
    return text

# Apply with progress tracking
tqdm.pandas(desc="Preprocessing text")
df['cleaned_text'] = df['text'].progress_apply(preprocess_text)
```

#### Feature Engineering

```python
def create_text_features(df, text_column='text'):
    """Create basic text features for analysis."""
    
    # Length features
    df['char_count'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    df['sentence_count'] = df[text_column].str.count(r'[.!?]+')
    
    # Readability features
    df['avg_word_length'] = df['char_count'] / df['word_count'].replace(0, 1)
    df['words_per_sentence'] = df['word_count'] / df['sentence_count'].replace(0, 1)
    
    print("📊 Created text features:")
    for col in ['char_count', 'word_count', 'avg_word_length']:
        print(f"   {col}: {df[col].describe()}")
    
    return df
```

## 🔧 Model Development Standards

### Experiment Configuration

```python
@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    
    # Data settings
    dataset_size: int = 5000  # For quick demos
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # Model settings
    max_features: int = 5000
    max_length: int = 100
    embedding_dim: int = 100
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    
    # Training settings
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    
    # Performance settings
    use_subset: bool = True  # For educational purposes
    show_progress: bool = True
    save_plots: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < self.val_size < 1, "val_size must be between 0 and 1"
        assert self.epochs > 0, "epochs must be positive"
        
        if self.use_subset and self.dataset_size > 10000:
            print("⚠️  Large dataset_size with use_subset=True. Consider reducing for speed.")
```

### Model Training with Monitoring

```python
def train_with_monitoring(model, train_loader, val_loader, config):
    """Train model with comprehensive monitoring."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (data, target) in enumerate(progress_bar):
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{np.mean(train_losses):.4f}"
                })
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}: Train Loss: {np.mean(train_losses):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    return history
```

## 📈 Visualization and Analysis

### Training Progress Visualization

```python
def plot_training_history(history, save_path=None):
    """Create comprehensive training visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss', alpha=0.8)
    axes[0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', alpha=0.8)
    axes[0].set_title('Model Loss Over Time')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['val_acc'], 'go-', label='Validation Accuracy', alpha=0.8)
    axes[1].set_title('Model Accuracy Over Time')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add annotations for best performance
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    axes[1].annotate(f'Best: {best_acc:.3f} at epoch {best_epoch}',
                    xy=(best_epoch, best_acc),
                    xytext=(best_epoch + 1, best_acc - 0.02),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()
```

### Model Evaluation

```python
def comprehensive_evaluation(model, test_loader, label_names=None):
    """Perform comprehensive model evaluation."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            output = model(data)
            predictions = torch.sigmoid(output).round()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"📊 Model Performance Summary:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Correct predictions: {sum(y_true == y_pred)}/{len(y_true)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names or ['Negative', 'Positive'],
                yticklabels=label_names or ['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'targets': y_true
    }
```

## 🛡️ Error Handling and Robustness

### Graceful Failure Handling

```python
def safe_operation(func, *args, fallback=None, error_msg="Operation failed", **kwargs):
    """Execute function with graceful error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️  {error_msg}: {str(e)}")
        if fallback is not None:
            print(f"🔄 Using fallback: {fallback}")
            return fallback
        else:
            print("❌ No fallback available")
            raise
```

### Memory Management

```python
def check_memory_usage():
    """Monitor memory usage and provide warnings."""
    import psutil
    
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent_used = memory.percent
    
    print(f"💾 Memory usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({percent_used:.1f}%)")
    
    if percent_used > 80:
        print("⚠️  High memory usage detected!")
        print("💡 Consider:")
        print("   - Using smaller dataset")
        print("   - Reducing batch size") 
        print("   - Processing in chunks")
    
    return percent_used < 90  # Return False if memory critical
```

### Data Quality Checks

```python
def validate_text_data(df, text_column, target_column=None):
    """Comprehensive text data validation."""
    
    issues = []
    
    # Check for missing values
    missing_text = df[text_column].isna().sum()
    if missing_text > 0:
        issues.append(f"Missing text values: {missing_text}")
    
    # Check for empty strings
    empty_text = (df[text_column] == "").sum()
    if empty_text > 0:
        issues.append(f"Empty text values: {empty_text}")
    
    # Check text length distribution
    lengths = df[text_column].str.len()
    if lengths.min() < 10:
        issues.append(f"Very short texts detected (min: {lengths.min()} characters)")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=[text_column]).sum()
    if duplicates > 0:
        issues.append(f"Duplicate texts: {duplicates}")
    
    # Check target distribution if provided
    if target_column:
        target_dist = df[target_column].value_counts()
        if len(target_dist) < 2:
            issues.append("Only one target class found")
        elif target_dist.min() / target_dist.max() < 0.1:
            issues.append("Severe class imbalance detected")
    
    # Report results
    if issues:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Data quality checks passed")
    
    return len(issues) == 0
```

## 🔄 Reproducibility Standards

### Seed Management

```python
def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seeds set to {seed} for reproducibility")
```

### Experiment Logging

```python
def log_experiment(config, results, model_path=None):
    """Log experiment details for reproducibility."""
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'results': results,
        'model_path': model_path,
        'git_commit': get_git_commit(),  # If in git repo
        'python_version': sys.version,
        'package_versions': get_package_versions()
    }
    
    # Save to JSON
    log_file = f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('experiments', exist_ok=True)
    
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2, default=str)
    
    print(f"📝 Experiment logged to {log_file}")
    return log_file
```

## 🎯 Educational Integration

### Progress Tracking for Students

```python
def create_learning_checkpoint(stage, data, message):
    """Create checkpoints for student learning."""
    
    checkpoint = {
        'stage': stage,
        'timestamp': datetime.now(),
        'data_shape': data.shape if hasattr(data, 'shape') else None,
        'message': message
    }
    
    print(f"🎓 Learning Checkpoint: {stage}")
    print(f"   {message}")
    if checkpoint['data_shape']:
        print(f"   Data shape: {checkpoint['data_shape']}")
    
    return checkpoint
```

### Interactive Learning Elements

```python
def create_interactive_demo(model, tokenizer, max_length=100):
    """Create interactive demo for model testing."""
    
    def predict_sentiment(text):
        """Predict sentiment for user input."""
        
        # Preprocess
        processed = preprocess_text(text)
        if not processed:
            return "Text too short or invalid"
        
        # Tokenize and predict
        tokens = tokenizer.encode(processed, max_length=max_length, truncation=True)
        input_tensor = torch.tensor([tokens])
        
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()
        
        sentiment = "Positive" if probability > 0.5 else "Negative"
        confidence = max(probability, 1 - probability)
        
        return f"{sentiment} (confidence: {confidence:.2%})"
    
    print("🎮 Interactive Demo Ready!")
    print("Enter text to analyze sentiment (or 'quit' to exit):")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
        
        result = predict_sentiment(user_input)
        print(f"Result: {result}\n")
```
