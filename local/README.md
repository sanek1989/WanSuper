# 🏠 WAN 2.5 Local Mode - Documentation

## 🎯 Overview

This directory contains the local mode implementation for WAN 2.5, featuring **Hybrid RAM+VRAM support** that enables running Flux Wan and WAN 2.5 models on GPUs with **as little as 4GB VRAM**.

## ✨ Key Features

### Hybrid Memory Management

- **4GB VRAM Support**: Run large AI models on modest hardware
- **Automatic Tensor Offloading**: When VRAM fills up, tensors automatically move to RAM
- **Smart Memory Allocation**: PyTorch intelligently manages GPU/CPU memory
- **Real-time Monitoring**: Track VRAM and RAM usage in the interface
- **Flexible Backend Switching**: Choose between GPU-only, Hybrid, or CPU-only modes

### How It Works

When you run models in **Hybrid RAM+VRAM mode**:

1. **Initial Load**: Model layers are loaded to GPU VRAM
2. **VRAM Management**: When VRAM reaches capacity, PyTorch automatically swaps inactive tensors to system RAM
3. **Active Processing**: Frequently-used tensors remain in VRAM for fast computation
4. **Dynamic Swapping**: Data moves between VRAM and RAM as needed during inference
5. **Automatic Cleanup**: Memory is released after processing completes

### Memory Requirements

| Configuration | VRAM | RAM | Speed | Recommended For |
|--------------|------|-----|-------|----------------|
| **GPU Only** | 12GB+ | 8GB+ | 🚀 Fast | RTX 3060 12GB, RTX 4070+ |
| **Hybrid (Recommended)** | 4GB+ | 16GB+ | 🚶 Moderate | RTX 3050, GTX 1650, Budget GPUs |
| **CPU Only** | N/A | 16GB+ | 🐢 Slow | No GPU available |

## 📦 Installation

### Prerequisites

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gradio psutil accelerate
```

### Additional Dependencies

For optimal memory management:

```bash
pip install accelerate  # For advanced tensor offloading
pip install bitsandbytes  # For 8-bit quantization (optional)
```

## 🚀 Quick Start

### Running Local Mode

```python
from local.local_interface import create_local_interface

# Create and launch the interface
interface = create_local_interface()
interface.launch()
```

### Configuration Examples

#### 1. Hybrid Mode (4GB GPU)

```python
import torch

# Configure for 4GB GPU
torch.cuda.set_per_process_memory_fraction(0.75, device=0)
device = torch.device('cuda')

print(f"VRAM limit set to 75% ({0.75 * torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
```

#### 2. GPU-Only Mode (12GB+ GPU)

```python
import torch

# Use full VRAM for maximum performance
torch.cuda.set_per_process_memory_fraction(0.95, device=0)
device = torch.device('cuda')
```

#### 3. CPU-Only Mode (No GPU)

```python
import torch

device = torch.device('cpu')
print("Running on CPU - slower but works without GPU")
```

## 💻 PyTorch Memory Management Best Practices

### 1. Setting VRAM Limits

Prevent Out of Memory (OOM) errors by limiting VRAM usage:

```python
import torch

# Reserve 20% VRAM for system/other apps
torch.cuda.set_per_process_memory_fraction(0.8, device=0)
```

### 2. Manual Memory Management

```python
import torch

# Clear unused cached memory
torch.cuda.empty_cache()

# Wait for all GPU operations to complete
torch.cuda.synchronize()

# Monitor memory usage
print(f"Allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
```

### 3. Device Selection

```python
import torch

# Automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors to device
model = model.to(device)
input_data = input_data.to(device)
```

### 4. Gradient Checkpointing

Save VRAM during training/fine-tuning:

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Trades compute for memory - slower but uses less VRAM
```

### 5. Mixed Precision Training

Reduce memory usage with automatic mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🔄 Advanced: Tensor Offloading with Accelerate

### Method 1: Automatic Device Mapping

```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

# Initialize model structure without loading weights
with init_empty_weights():
    model = FluxWanModel.from_pretrained(model_path)

# Load with automatic device mapping
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=model_path,
    device_map="auto",  # Automatically split across GPU/CPU
    offload_folder="offload",  # Temporary storage for offloaded weights
    offload_state_dict=True,  # Enable state dict offloading
    max_memory={0: "3GB", "cpu": "16GB"}  # Memory limits per device
)
```

### Method 2: CPU Offload Decorator

```python
from accelerate import cpu_offload
import torch

@cpu_offload(model, execution_device='cuda:0')
def generate_video(model, prompt, **kwargs):
    """Model automatically moves between GPU and CPU."""
    with torch.no_grad():
        video = model.generate(prompt, **kwargs)
    return video

# Model layers move to GPU only when needed
result = generate_video(model, "A beautiful sunset")
```

### Method 3: Sequential Processing with Memory Clearing

```python
import torch

def process_frames_sequentially(model, frames, device='cuda'):
    """
    Process video frames one at a time to minimize VRAM usage.
    """
    results = []
    
    for i, frame in enumerate(frames):
        # Move frame to GPU
        frame = frame.to(device)
        
        # Process
        with torch.no_grad():
            result = model(frame)
        
        # Move result back to CPU to free VRAM
        results.append(result.cpu())
        
        # Clean up
        del frame, result
        torch.cuda.empty_cache()
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")
    
    return torch.cat(results)
```

### Method 4: Checkpoint Sequential

Split model into segments and checkpoint:

```python
from torch.utils.checkpoint import checkpoint_sequential

# Split model into 4 checkpointed segments
num_segments = 4

# Each segment is computed one at a time, saving memory
output = checkpoint_sequential(model, num_segments, input_tensor)
```

## 📊 Performance Optimization Tips

### For 4GB GPUs:

1. **VRAM Fraction**: Set to 0.70-0.80 (70-80%)
2. **RAM**: Ensure 16GB+ available
3. **Batch Size**: Use 1 for minimal memory usage
4. **Resolution**: Start with lower resolutions
5. **Close Background Apps**: Free up both VRAM and RAM

### For 6-8GB GPUs:

1. **VRAM Fraction**: Set to 0.85-0.90 (85-90%)
2. **RAM**: 12GB+ recommended
3. **Batch Size**: Can try 2-4
4. **Resolution**: Medium to high

### For 12GB+ GPUs:

1. **VRAM Fraction**: Set to 0.95 (95%)
2. **Mode**: Use GPU-only mode
3. **Batch Size**: 4-8 or higher
4. **Resolution**: Full quality

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory` error

**Solutions**:
1. Lower VRAM fraction: `torch.cuda.set_per_process_memory_fraction(0.65, device=0)`
2. Close other GPU applications (browsers, games, etc.)
3. Reduce batch size to 1
4. Ensure sufficient system RAM (16GB+)
5. Try CPU-only mode as fallback

### Slow Performance

**Symptoms**: Generation takes much longer than expected

**Causes & Solutions**:
1. **Excessive swapping**: Increase VRAM fraction or upgrade GPU
2. **CPU bottleneck**: Check CPU usage, close background apps
3. **Insufficient RAM**: Upgrade to 16GB+ RAM
4. **First run**: Initial model loading is always slower

### Model Won't Load

**Symptoms**: Errors during model initialization

**Solutions**:
1. Check available RAM: Need 12GB+ free
2. Verify model files integrity
3. Ensure PyTorch is installed correctly: `pip install torch --upgrade`
4. Check CUDA compatibility: `torch.cuda.is_available()`

### Interface Errors

**Symptoms**: Gradio interface fails to launch

**Solutions**:
1. Update dependencies: `pip install -r requirements.txt --upgrade`
2. Check port availability (default: 7860)
3. Review error messages in console
4. Ensure all imports are available

## 📈 Memory Monitoring

### Real-time Memory Status

The interface provides real-time monitoring:

```python
import torch
import psutil

def get_memory_status():
    """Get current memory usage."""
    
    # System RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.available / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB available")
    
    # GPU VRAM
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        vram_free = vram_total - vram_allocated
        
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_free:.1f}GB / {vram_total:.1f}GB available")
    else:
        print("GPU: Not available")

get_memory_status()
```

## 🔧 Configuration Reference

### Environment Variables

```bash
# Limit PyTorch CUDA memory caching
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TensorFloat-32 (faster on Ampere GPUs)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

### Configuration Options

| Option | Values | Description |
|--------|--------|-------------|
| `memory_mode` | `gpu_only`, `hybrid`, `cpu_only` | Processing backend |
| `vram_fraction` | 0.5 - 0.95 | Fraction of VRAM to use |
| `offload_folder` | Path string | Temp folder for offloaded weights |
| `device_map` | `auto`, `sequential`, custom dict | How to distribute model layers |

## 📚 Additional Resources

### Official Documentation

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [Accelerate Library](https://huggingface.co/docs/accelerate/)
- [Model Optimization Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [WAN 2.5 Information](https://help.aliyun.com/zh/dashscope/)

### Tutorials

- [Large Model Inference on Limited Hardware](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [Memory-Efficient Training](https://pytorch.org/docs/stable/notes/large_scale_deployments.html)
- [Gradient Checkpointing Tutorial](https://pytorch.org/docs/stable/checkpoint.html)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project follows the same license as the main WanSuper repository.

---

**Made with ❤️ by The Angel Studio**

🌟 [Support The Angel Studio on Boosty](https://boosty.to/the_angel)
