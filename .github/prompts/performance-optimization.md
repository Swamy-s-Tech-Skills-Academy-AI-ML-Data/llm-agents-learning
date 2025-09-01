# Performance Optimization and Scaling Guide

## 🎯 Purpose

This prompt guides AI assistants in optimizing performance for the NLP learning repository while maintaining educational value and Windows compatibility.

## ⚡ Performance Optimization Strategies

### Memory Optimization

#### Efficient Data Loading

```python
def load_data_efficiently(file_path, chunk_size=10000, max_rows=None):
    """Load large datasets efficiently with memory management."""
    
    # Check file size first
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 100:
        print("⚠️  Large file detected. Using chunked loading...")
        
        # Read in chunks
        chunks = []
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Loading chunks"):
            if max_rows and len(chunks) * chunk_size >= max_rows:
                chunk = chunk.head(max_rows - len(chunks) * chunk_size)
                chunks.append(chunk)
                break
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"✅ Loaded {len(df):,} rows efficiently")
        
    else:
        # Direct loading for smaller files
        nrows = max_rows if max_rows else None
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"✅ Loaded {len(df):,} rows directly")
    
    return df
```

#### Memory-Efficient Text Processing

```python
def process_text_in_batches(texts, batch_size=1000, processor_func=None):
    """Process large text collections in memory-efficient batches."""
    
    if processor_func is None:
        processor_func = lambda x: x.lower().strip()
    
    processed = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), 
                  desc="Processing text batches",
                  total=total_batches):
        
        batch = texts[i:i + batch_size]
        batch_processed = [processor_func(text) for text in batch]
        processed.extend(batch_processed)
        
        # Periodically free memory
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    return processed
```

### Computational Optimization

#### Smart Subset Selection

```python
def create_performance_subset(df, target_col=None, subset_size=5000, 
                            stratify=True, random_state=42):
    """Create representative subset for faster experimentation."""
    
    print(f"🔀 Creating subset of {subset_size:,} from {len(df):,} samples")
    
    if subset_size >= len(df):
        print("ℹ️  Subset size >= dataset size. Using full dataset.")
        return df
    
    if stratify and target_col and target_col in df.columns:
        # Stratified sampling to maintain class distribution
        subset = df.groupby(target_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), subset_size // df[target_col].nunique()),
                             random_state=random_state)
        ).reset_index(drop=True)
        
        print("📊 Class distribution in subset:")
        print(subset[target_col].value_counts())
        
    else:
        # Random sampling
        subset = df.sample(n=subset_size, random_state=random_state).reset_index(drop=True)
    
    print(f"✅ Created subset with {len(subset):,} samples")
    return subset
```

#### Parallel Processing

```python
def parallel_text_processing(texts, func, n_jobs=-1, batch_size=1000):
    """Process text data in parallel for speed improvement."""
    
    from multiprocessing import Pool, cpu_count
    import concurrent.futures
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    print(f"🚀 Processing {len(texts):,} texts using {n_jobs} cores")
    
    # Split into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    def process_batch(batch):
        return [func(text) for text in batch]
    
    # Process in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(
            executor.map(process_batch, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Flatten results
    processed = [item for sublist in results for item in sublist]
    print(f"✅ Parallel processing completed")
    
    return processed
```

### Model Training Optimization

#### Smart Training Strategies

```python
def progressive_training(model, train_loader, val_loader, config):
    """Train with progressive complexity for educational value."""
    
    # Start with small subset for quick validation
    print("🎯 Phase 1: Quick validation (1 epoch, small batch)")
    quick_config = copy.deepcopy(config)
    quick_config.epochs = 1
    quick_config.batch_size = min(config.batch_size, 16)
    
    # Quick training to validate pipeline
    quick_history = train_model(model, train_loader, val_loader, quick_config)
    
    if quick_history['val_acc'][-1] < 0.6:
        print("⚠️  Quick validation shows poor performance. Check data/model.")
        return quick_history
    
    print("✅ Pipeline validated. Starting full training...")
    
    # Full training
    print(f"🎯 Phase 2: Full training ({config.epochs} epochs)")
    full_history = train_model(model, train_loader, val_loader, config)
    
    return full_history
```

#### GPU Optimization

```python
def optimize_for_device(model, device_preference='auto'):
    """Optimize model for available hardware."""
    
    if device_preference == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            
            # GPU-specific optimizations
            torch.backends.cudnn.benchmark = True
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
        else:
            device = torch.device('cpu')
            print("💻 Using CPU (GPU not available)")
            
            # CPU-specific optimizations
            torch.set_num_threads(os.cpu_count())
            
    else:
        device = torch.device(device_preference)
    
    model = model.to(device)
    return model, device
```

## 📊 Performance Monitoring

### Training Performance Tracking

```python
class PerformanceMonitor:
    """Monitor training performance and resource usage."""
    
    def __init__(self):
        self.metrics = {
            'epoch_times': [],
            'memory_usage': [],
            'gpu_usage': [] if torch.cuda.is_available() else None,
            'loss_progression': [],
            'learning_rate': []
        }
        self.start_time = None
    
    def start_epoch(self):
        """Mark start of epoch."""
        self.start_time = time.time()
        
        # Record memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
            self.metrics['memory_usage'].append(memory_mb)
    
    def end_epoch(self, loss, lr):
        """Mark end of epoch and record metrics."""
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.metrics['epoch_times'].append(epoch_time)
        
        self.metrics['loss_progression'].append(loss)
        self.metrics['learning_rate'].append(lr)
        
        # Print performance update
        if len(self.metrics['epoch_times']) > 0:
            avg_time = np.mean(self.metrics['epoch_times'][-5:])  # Last 5 epochs
            print(f"⏱️  Avg epoch time: {avg_time:.1f}s")
    
    def plot_performance(self):
        """Plot performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Epoch times
        axes[0, 0].plot(self.metrics['epoch_times'])
        axes[0, 0].set_title('Training Speed')
        axes[0, 0].set_ylabel('Seconds per Epoch')
        
        # Memory usage
        if self.metrics['memory_usage']:
            axes[0, 1].plot(self.metrics['memory_usage'])
            axes[0, 1].set_title('Memory Usage')
            axes[0, 1].set_ylabel('Memory (MB)')
        
        # Loss progression
        axes[1, 0].plot(self.metrics['loss_progression'])
        axes[1, 0].set_title('Loss Progression')
        axes[1, 0].set_ylabel('Loss')
        
        # Learning rate
        axes[1, 1].plot(self.metrics['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()
        plt.show()
```

### Resource Usage Monitoring

```python
def monitor_system_resources():
    """Monitor and report system resource usage."""
    
    import psutil
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1e9
    memory_used_percent = memory.percent
    
    # Disk usage
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / 1e9
    
    print("🖥️  System Resources:")
    print(f"   CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    print(f"   Memory: {memory_used_percent:.1f}% of {memory_gb:.1f} GB")
    print(f"   Disk free: {disk_free_gb:.1f} GB")
    
    # GPU if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_percent = (gpu_memory / gpu_total) * 100
        
        print(f"   GPU Memory: {gpu_percent:.1f}% of {gpu_total:.1f} GB")
    
    # Return warnings for high usage
    warnings = []
    if cpu_percent > 90:
        warnings.append("High CPU usage detected")
    if memory_used_percent > 85:
        warnings.append("High memory usage detected")
    if disk_free_gb < 1:
        warnings.append("Low disk space")
    
    return warnings
```

## 🚀 Scaling Strategies

### Data Scaling

#### Incremental Learning Approach

```python
def incremental_learning_demo(full_dataset, model_class, config):
    """Demonstrate incremental learning with growing datasets."""
    
    sizes = [1000, 2500, 5000, 10000, len(full_dataset)]
    results = []
    
    for size in sizes:
        if size > len(full_dataset):
            size = len(full_dataset)
        
        print(f"\n🎯 Training with {size:,} samples")
        
        # Create subset
        subset = full_dataset.head(size)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(subset, config)
        
        # Train model
        model = model_class(config)
        history = train_model(model, train_loader, val_loader, config)
        
        # Record results
        best_acc = max(history['val_acc'])
        results.append({
            'dataset_size': size,
            'best_accuracy': best_acc,
            'training_time': sum(history.get('epoch_times', [0]))
        })
        
        print(f"✅ Best accuracy: {best_acc:.4f}")
    
    # Plot scaling results
    plot_scaling_results(results)
    return results

def plot_scaling_results(results):
    """Plot how performance scales with dataset size."""
    
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy vs dataset size
    ax1.plot(df_results['dataset_size'], df_results['best_accuracy'], 'bo-')
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Best Accuracy')
    ax1.set_title('Accuracy vs Dataset Size')
    ax1.grid(True, alpha=0.3)
    
    # Training time vs dataset size
    ax2.plot(df_results['dataset_size'], df_results['training_time'], 'ro-')
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time vs Dataset Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Model Scaling

#### Progressive Model Complexity

```python
def compare_model_complexities(data_loader, config):
    """Compare different model complexities for educational insight."""
    
    model_configs = [
        {'hidden_dim': 32, 'num_layers': 1, 'name': 'Simple'},
        {'hidden_dim': 64, 'num_layers': 1, 'name': 'Medium'},
        {'hidden_dim': 128, 'num_layers': 2, 'name': 'Complex'},
        {'hidden_dim': 256, 'num_layers': 3, 'name': 'Very Complex'}
    ]
    
    results = []
    
    for model_config in model_configs:
        print(f"\n🧠 Testing {model_config['name']} model:")
        print(f"   Hidden dim: {model_config['hidden_dim']}")
        print(f"   Layers: {model_config['num_layers']}")
        
        # Update config
        test_config = copy.deepcopy(config)
        test_config.hidden_dim = model_config['hidden_dim']
        test_config.num_layers = model_config['num_layers']
        
        # Train model
        model = create_model(test_config)
        history = train_model(model, data_loader, test_config)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        results.append({
            'name': model_config['name'],
            'parameters': param_count,
            'best_accuracy': max(history['val_acc']),
            'training_time': sum(history.get('epoch_times', [0]))
        })
        
        print(f"   Parameters: {param_count:,}")
        print(f"   Best accuracy: {max(history['val_acc']):.4f}")
    
    # Compare results
    df_comparison = pd.DataFrame(results)
    print("\n📊 Model Complexity Comparison:")
    print(df_comparison.to_string(index=False))
    
    return df_comparison
```

## 🎓 Educational Performance Considerations

### Interactive Performance Demos

```python
def performance_comparison_demo():
    """Interactive demo showing performance trade-offs."""
    
    print("🎮 Performance Comparison Demo")
    print("=" * 40)
    
    # Let user choose between approaches
    print("Choose processing approach:")
    print("1. Quick demo (1K samples, fast)")
    print("2. Medium demo (5K samples, moderate)")
    print("3. Full processing (50K samples, slow)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    size_map = {'1': 1000, '2': 5000, '3': 50000}
    size = size_map.get(choice, 1000)
    
    # Estimate time
    time_estimates = {1000: "~30 seconds", 5000: "~2 minutes", 50000: "~15 minutes"}
    
    print(f"\n⏱️  Estimated time: {time_estimates.get(size, 'Unknown')}")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Demo cancelled.")
        return
    
    # Run demo with chosen size
    start_time = time.time()
    # ... actual processing here ...
    end_time = time.time()
    
    print(f"✅ Completed in {end_time - start_time:.1f} seconds")
    print(f"📊 Processed {size:,} samples")
```

### Performance Education

```python
def explain_performance_concepts():
    """Explain key performance concepts to beginners."""
    
    concepts = {
        "Batch Size": {
            "description": "Number of samples processed together",
            "trade_off": "Larger = faster training, more memory usage",
            "recommendation": "Start with 32, increase if memory allows"
        },
        "Epochs": {
            "description": "Number of complete passes through the data",
            "trade_off": "More = better learning, longer training time",
            "recommendation": "Start with 5, monitor for overfitting"
        },
        "Model Size": {
            "description": "Number of parameters in the neural network",
            "trade_off": "Larger = more capacity, slower training",
            "recommendation": "Start simple, increase complexity gradually"
        },
        "Data Size": {
            "description": "Number of training samples",
            "trade_off": "More = better performance, longer training",
            "recommendation": "Use subsets for experimentation"
        }
    }
    
    print("📚 Performance Concepts Guide:")
    print("=" * 50)
    
    for concept, details in concepts.items():
        print(f"\n🔹 {concept}")
        print(f"   What: {details['description']}")
        print(f"   Trade-off: {details['trade_off']}")
        print(f"   💡 Tip: {details['recommendation']}")
```

## ⚠️ Performance Pitfalls to Avoid

### Memory Leaks

```python
def prevent_memory_leaks():
    """Common patterns to prevent memory leaks in PyTorch."""
    
    # Clear GPU cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Detach tensors when storing for plotting
    losses_for_plot = [loss.detach().cpu().item() for loss in loss_list]
    
    # Use context managers for evaluation
    with torch.no_grad():
        # evaluation code here
        pass
    
    # Clear variables explicitly when done
    del large_tensor
    gc.collect()
```

### Inefficient Data Loading

```python
# ❌ Inefficient approach
def load_data_inefficiently():
    """Example of what NOT to do."""
    
    # Loading entire dataset into memory at once
    df = pd.read_csv("huge_file.csv")  # Memory explosion
    
    # Converting to PyTorch tensors all at once
    all_data = torch.tensor(df.values)  # Double memory usage
    
    # No batching or streaming
    return all_data

# ✅ Efficient approach
def load_data_efficiently():
    """Example of efficient data loading."""
    
    # Use generators and lazy loading
    def data_generator():
        for chunk in pd.read_csv("huge_file.csv", chunksize=1000):
            yield process_chunk(chunk)
    
    # Use PyTorch DataLoader for automatic batching
    dataset = CustomDataset(data_generator())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return loader
```
