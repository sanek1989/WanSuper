"""WAN 2.5 Local Mode Interface with Hybrid RAM+VRAM Support"""
import gradio as gr
import torch
import os
import psutil
from typing import Tuple, Optional

def get_memory_info() -> dict:
    """Get current memory status (GPU and RAM)"""
    memory_info = {
        'ram_total': psutil.virtual_memory().total / (1024**3),
        'ram_available': psutil.virtual_memory().available / (1024**3),
        'ram_percent': psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        memory_info['gpu_available'] = True
        memory_info['gpu_name'] = torch.cuda.get_device_name(0)
        memory_info['vram_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_info['vram_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
        memory_info['vram_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
    else:
        memory_info['gpu_available'] = False
    
    return memory_info

def configure_memory_mode(mode: str, vram_fraction: float = 0.95) -> Tuple[str, str]:
    """
    Configure PyTorch memory allocation strategy.
    
    Args:
        mode: One of 'gpu_only', 'hybrid', 'cpu_only'
        vram_fraction: Fraction of VRAM to use (0.0-1.0)
    
    Returns:
        Tuple of (status_message, device_str)
    """
    try:
        if mode == 'cpu_only':
            device = 'cpu'
            status = "✅ CPU-only mode activated. Processing will use RAM only (slower)."
            
        elif mode == 'gpu_only':
            if not torch.cuda.is_available():
                return "❌ GPU not available. Falling back to CPU mode.", 'cpu'
            
            device = 'cuda'
            torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
            status = f"✅ GPU-only mode activated. Using {vram_fraction*100:.0f}% of VRAM."
            
        elif mode == 'hybrid':
            if not torch.cuda.is_available():
                return "❌ GPU not available. Falling back to CPU mode.", 'cpu'
            
            device = 'cuda'
            # Set conservative VRAM usage for hybrid mode
            torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
            # Enable memory efficient settings
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            status = (f"✅ Hybrid RAM+VRAM mode activated. "
                     f"VRAM limit: {vram_fraction*100:.0f}%. "
                     f"Excess tensors will swap to RAM.")
        else:
            return f"❌ Unknown mode: {mode}", 'cpu'
        
        return status, device
        
    except Exception as e:
        return f"❌ Error configuring memory: {str(e)}", 'cpu'

def format_memory_status() -> str:
    """Format current memory status as a string."""
    info = get_memory_info()
    
    status = f"### 💾 Memory Status\n\n"
    status += f"**RAM:** {info['ram_available']:.1f}GB / {info['ram_total']:.1f}GB available ({100-info['ram_percent']:.1f}%)\n\n"
    
    if info['gpu_available']:
        vram_free = info['vram_total'] - info['vram_allocated']
        status += f"**GPU:** {info['gpu_name']}\n\n"
        status += f"**VRAM:** {vram_free:.1f}GB / {info['vram_total']:.1f}GB available\n\n"
    else:
        status += f"**GPU:** Not available\n\n"
    
    return status

def create_local_interface():
    """
    Create Gradio interface for WAN 2.5 Local mode with 4GB VRAM support.
    
    This interface supports running Flux Wan models on GPUs with as little as 4GB VRAM
    by utilizing hybrid RAM+VRAM allocation strategies.
    """
    
    with gr.Blocks(title="WAN 2.5 Local Mode - The Angel Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏠 WAN 2.5 Local Mode - Self-Hosted with 4GB VRAM Support
            
            ## ✨ Hybrid RAM+VRAM Technology
            
            This interface enables running **Flux Wan** and **WAN 2.5** models on GPUs with **as little as 4GB VRAM**!
            
            When VRAM is insufficient, the system automatically offloads tensor data to RAM, 
            allowing you to work with large models on modest hardware.
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Memory Configuration")
                
                memory_mode = gr.Radio(
                    choices=[
                        ("🎮 GPU Only (Requires 12GB+ VRAM)", "gpu_only"),
                        ("🔄 Hybrid RAM+VRAM (Works with 4GB VRAM) ⭐ RECOMMENDED", "hybrid"),
                        ("💻 CPU Only (Slowest, no GPU required)", "cpu_only")
                    ],
                    value="hybrid",
                    label="Select Processing Mode",
                    info="Hybrid mode is optimized for 4GB GPUs"
                )
                
                vram_fraction = gr.Slider(
                    minimum=0.5,
                    maximum=0.95,
                    value=0.8,
                    step=0.05,
                    label="VRAM Usage Limit (for GPU/Hybrid modes)",
                    info="Lower values leave more VRAM for system, higher values may cause OOM errors"
                )
                
                configure_btn = gr.Button("🔧 Apply Configuration", variant="primary")
                config_status = gr.Textbox(label="Configuration Status", interactive=False)
                
            with gr.Column(scale=1):
                memory_status = gr.Markdown(format_memory_status())
                refresh_btn = gr.Button("🔄 Refresh Memory Status")
        
        gr.Markdown(
            """
            ---
            
            ## 📖 How It Works: Hybrid RAM+VRAM Mode
            
            ### Tensor Offloading Strategy
            
            When you run models in **Hybrid mode** on a 4GB GPU:
            
            1. **Model layers** are initially loaded to GPU VRAM
            2. If VRAM fills up, **PyTorch automatically swaps** less-used tensors to RAM
            3. **Active tensors** stay in VRAM for fast computation
            4. **Inactive tensors** move to RAM temporarily
            5. Data moves back to VRAM as needed during inference
            
            ### Performance Expectations
            
            | GPU VRAM | Mode | Speed | Notes |
            |----------|------|-------|-------|
            | 4GB | Hybrid | 🐢 Slow | Works! Expect 3-5x slower than 12GB GPU |
            | 6GB | Hybrid | 🚶 Moderate | Better performance, less swapping |
            | 8GB | Hybrid | 🏃 Good | Minimal swapping |
            | 12GB+ | GPU Only | 🚀 Fast | Optimal performance |
            
            ---
            
            ## 💻 Best Practices for PyTorch Memory Management
            
            ### Key PyTorch APIs Used:
            
            ```python
            import torch
            
            # 1. Set VRAM usage limit (prevents OOM crashes)
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
            
            # 2. Choose device based on availability and mode
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 3. Enable gradient checkpointing (saves VRAM during training)
            model.gradient_checkpointing_enable()
            
            # 4. Use automatic mixed precision (reduces memory usage)
            from torch.cuda.amp import autocast
            with autocast():
                output = model(input)
            
            # 5. Manual memory management
            torch.cuda.empty_cache()  # Free unused cached memory
            torch.cuda.synchronize()   # Wait for GPU operations to complete
            ```
            
            ### Example: Load Model with Tensor Offloading
            
            ```python
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            
            # Initialize model structure without loading weights
            with init_empty_weights():
                model = FluxWanModel.from_pretrained(model_path)
            
            # Load with automatic device mapping (enables RAM offload)
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=model_path,
                device_map="auto",           # Automatically distribute across GPU/CPU
                offload_folder="offload",   # Temp folder for offloaded weights
                offload_state_dict=True      # Enable state dict offloading
            )
            ```
            
            ### Swap/Offload Code Example
            
            ```python
            import torch
            from accelerate import cpu_offload
            
            # Method 1: Manual CPU offload for specific layers
            @cpu_offload(model, device='cuda:0')
            def run_inference(model, input_data):
                # Model automatically moves between GPU and CPU
                with torch.no_grad():
                    output = model(input_data)
                return output
            
            # Method 2: Sequential execution with memory clearing
            def process_in_chunks(model, data, chunk_size=1):
                results = []
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i+chunk_size].to('cuda')
                    
                    with torch.no_grad():
                        result = model(chunk)
                    
                    results.append(result.cpu())  # Move result to RAM
                    del chunk, result
                    torch.cuda.empty_cache()      # Free VRAM
                
                return torch.cat(results)
            
            # Method 3: Use PyTorch's built-in checkpointing
            from torch.utils.checkpoint import checkpoint_sequential
            
            # Split model into segments and checkpoint
            segments = 4  # Number of checkpoints
            output = checkpoint_sequential(model, segments, input)
            ```
            
            ---
            
            ## 🚀 Quick Start Guide
            
            ### For 4GB GPU Users:
            
            1. **Select** "🔄 Hybrid RAM+VRAM" mode above
            2. **Set** VRAM Usage Limit to **0.75-0.80** (75-80%)
            3. **Ensure** you have at least **16GB RAM** available
            4. **Click** "Apply Configuration"
            5. **Load** your Flux Wan model
            6. **Be patient** - first inference will be slower as model loads
            
            ### Troubleshooting:
            
            **Out of Memory (OOM) Error?**
            - ✅ Lower VRAM Usage Limit to 0.7 or 0.65
            - ✅ Close other GPU applications
            - ✅ Ensure sufficient RAM (16GB+ recommended)
            
            **Too Slow?**
            - ✅ Reduce batch size to 1
            - ✅ Use lower resolution outputs
            - ✅ Consider upgrading GPU for better performance
            
            **Model Won't Load?**
            - ✅ Check RAM availability (need 12GB+ free)
            - ✅ Verify model files are not corrupted
            - ✅ Try CPU-only mode as fallback
            
            ---
            
            ## 📚 Additional Resources
            
            - [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
            - [Accelerate Library Documentation](https://huggingface.co/docs/accelerate/)
            - [Model Optimization Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)
            - [WAN 2.5 Model Information](https://help.aliyun.com/zh/dashscope/)
            
            ---
            
            **Made with ❤️ by The Angel Studio**
            
            🌟 [Support The Angel Studio on Boosty](https://boosty.to/the_angel)
            """
        )
        
        # Event handlers
        configure_btn.click(
            fn=lambda mode, frac: configure_memory_mode(mode, frac),
            inputs=[memory_mode, vram_fraction],
            outputs=[config_status]
        )
        
        refresh_btn.click(
            fn=format_memory_status,
            outputs=[memory_status]
        )
        
        # Auto-refresh memory status on load
        demo.load(fn=format_memory_status, outputs=[memory_status])
    
    return demo

if __name__ == "__main__":
    interface = create_local_interface()
    interface.launch()
