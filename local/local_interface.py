"""WAN 2.5 Local Mode Interface with Hybrid RAM+VRAM Support"""
import gradio as gr
import os
from typing import Tuple, Optional

# Lazy-friendly availability flags (so UI can load without heavy deps)
TORCH_AVAILABLE = True
PSUTIL_AVAILABLE = True

try:
    import torch  # type: ignore
except Exception:
    TORCH_AVAILABLE = False

try:
    import psutil  # type: ignore
except Exception:
    PSUTIL_AVAILABLE = False

# ---------------------- Memory helpers (kept, safe) ----------------------
def _safe_psutil_vm():
    if not PSUTIL_AVAILABLE:
        # Minimal fallback without psutil
        return type("VM", (), {
            "total": 0,
            "available": 0,
            "percent": 0,
        })()
    # psutil is available, so we can import and use it
    import psutil
    return psutil.virtual_memory()

def get_memory_info() -> dict:
    """Get current memory status (GPU and RAM) without hard dependency on torch/psutil"""
    if not PSUTIL_AVAILABLE:
        # Return minimal info with warning
        memory_info = {
            'ram_total': 0.0,
            'ram_available': 0.0,
            'ram_percent': 0.0,
            'gpu_available': False,
            'psutil_warning': 'psutil не установлен, мониторинг памяти недоступен. Установите: pip install psutil'
        }
        return memory_info
    
    vm = _safe_psutil_vm()
    to_gib = lambda x: (x / (1024**3)) if isinstance(x, (int, float)) else 0.0
    
    memory_info = {
        'ram_total': to_gib(getattr(vm, 'total', 0)),
        'ram_available': to_gib(getattr(vm, 'available', 0)),
        'ram_percent': getattr(vm, 'percent', 0),
    }
    
    # Determine GPU availability lazily
    if TORCH_AVAILABLE:
        try:
            gpu_available = hasattr(torch, 'cuda') and torch.cuda.is_available()
        except Exception:
            gpu_available = False
    else:
        gpu_available = False
    
    if gpu_available:
        try:
            memory_info['gpu_available'] = True
            memory_info['gpu_name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            memory_info['vram_total'] = props.total_memory / (1024**3)
            memory_info['vram_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)
            memory_info['vram_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)
        except Exception:
            memory_info['gpu_available'] = True
            memory_info['gpu_name'] = 'CUDA device'
            memory_info['vram_total'] = 0.0
            memory_info['vram_allocated'] = 0.0
            memory_info['vram_reserved'] = 0.0
    else:
        memory_info['gpu_available'] = False
    
    return memory_info

def configure_memory_mode(mode: str, vram_fraction: float = 0.95) -> Tuple[str, str]:
    """
    Configure PyTorch memory allocation strategy lazily.
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
            if not TORCH_AVAILABLE:
                return ("❌ torch не установлен. GPU режим недоступен. Установите PyTorch: https://pytorch.org/get-started/locally/", 'cpu')
            if not torch.cuda.is_available():
                return "❌ GPU not available. Falling back to CPU mode.", 'cpu'
            device = 'cuda'
            try:
                torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
            except Exception:
                pass
            status = f"✅ GPU-only mode activated. Using {vram_fraction*100:.0f}% of VRAM."
        elif mode == 'hybrid':
            if not TORCH_AVAILABLE:
                return ("❌ torch не установлен. Гибридный режим недоступен. Установите PyTorch: https://pytorch.org/get-started/locally/", 'cpu')
            if not torch.cuda.is_available():
                return "❌ GPU not available. Falling back to CPU mode.", 'cpu'
            device = 'cuda'
            # Set conservative VRAM usage for hybrid mode
            try:
                torch.cuda.set_per_process_memory_fraction(vram_fraction, device=0)
                # Enable memory efficient settings
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except Exception:
                pass
            status = (f"✅ Hybrid RAM+VRAM mode activated. ")
        else:
            device = 'cpu'
            status = "ℹ️ Defaulting to CPU mode."
    except Exception as e:
        return f"❌ Failed to configure memory: {e}", 'cpu'
    
    return status, device

# ---------------------- UI helpers for model panels ----------------------
def model_panel_flux():
    desc = (
        "Flux — быстрый генеративный движок. Рекомендации: batch=1-2, 512-768px, CFG 4-7, "
        "FP16, автокаст. Требует CUDA для максимальной скорости."
    )
    with gr.Column(scale=7, min_width=420):
        gr.Markdown("## Flux • The Angel Studio")
        with gr.Group():
            flux_ckpt = gr.Textbox(label="Путь к чекпоинту", placeholder="/path/to/flux.ckpt")
            flux_steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
            flux_cfg = gr.Slider(0.0, 15.0, value=6.5, step=0.5, label="CFG")
            flux_res = gr.Slider(256, 1536, value=768, step=64, label="Resolution")
            flux_seed = gr.Number(value=42, precision=0, label="Seed")
        gr.Markdown(desc)
    return [flux_ckpt, flux_steps, flux_cfg, flux_res, flux_seed]

def model_panel_wan22():
    desc = (
        "Wan 2.2 — качественный детализатор. Рекомендации: batch=1, 768-1024px, CFG 6-9, "
        "xformers/attention оптимизации. Гибридный режим памяти поддерживается."
    )
    with gr.Column(scale=7, min_width=420):
        gr.Markdown("## Wan 2.2 • The Angel Studio")
        with gr.Group():
            wan_ckpt = gr.Textbox(label="Путь к модели", placeholder="/path/to/wan-2.2.safetensors")
            wan_steps = gr.Slider(1, 150, value=40, step=1, label="Steps")
            wan_cfg = gr.Slider(0.0, 20.0, value=7.5, step=0.5, label="CFG")
            wan_res = gr.Slider(256, 2048, value=1024, step=64, label="Resolution")
            wan_sampler = gr.Dropdown(["Euler", "DPM++ 2M Karras", "UniPC"], value="DPM++ 2M Karras", label="Sampler")
        gr.Markdown(desc)
    return [wan_ckpt, wan_steps, wan_cfg, wan_res, wan_sampler]

# ---------------------- Build UI ----------------------
def create_local_interface():
    angel_theme_css = """
    :root { --angel-bg: #0b0e14; --angel-panel: #121723; --angel-accent: #b38bfa; --angel-fg:#e7e9ee; }
    body, .gradio-container { background: var(--angel-bg) !important; color: var(--angel-fg) !important; }
    .angel-card { background: linear-gradient(180deg, #121723, #0f1420); border: 1px solid #252a36; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.35); }
    .angel-accent { color: var(--angel-accent) !important; }
    .angel-scroll { max-height: 420px; overflow-y: auto; padding-right: 6px; }
    .angel-scroll::-webkit-scrollbar { width: 8px; }
    .angel-scroll::-webkit-scrollbar-thumb { background: #2a3040; border-radius: 8px; }
    .angel-dd .wrap-inner { background: #141a29 !important; border: 1px solid #283049 !important; }
    .angel-dd .label { color: var(--angel-fg) !important; }
    """
    
    with gr.Blocks(css=angel_theme_css, title="WAN Local • The Angel Studio") as demo:
        with gr.Row():
            with gr.Column(scale=3, min_width=280, elem_classes=["angel-card"]):
                gr.Markdown("### Модель • <span class='angel-accent'>The Angel Studio</span>", elem_id="angel-title")
                
                # Styled Dropdown with only Flux and Wan 2.2
                model_select = gr.Dropdown(
                    choices=["Flux", "Wan 2.2"],
                    value="Flux",
                    label="Выбор нейросети",
                    filterable=False,
                    interactive=True,
                    elem_classes=["angel-dd", "angel-scroll"],
                )
                
                # Warning panel if torch/psutil missing
                missing_msgs = []
                if not TORCH_AVAILABLE:
                    missing_msgs.append("⚠️ PyTorch (torch) не найден. Установка: https://pytorch.org/get-started/locally/")
                if not PSUTIL_AVAILABLE:
                    missing_msgs.append("⚠️ psutil не найден. Установка: pip install psutil")
                if missing_msgs:
                    gr.Markdown("\n".join(missing_msgs))
                
                gr.Markdown("Масштабируйте список колесом — оформление под Angel Studio.")
                
                with gr.Accordion("Настройки памяти", open=False):
                    memory_mode = gr.Radio(["gpu_only", "hybrid", "cpu_only"], value="hybrid", label="Режим памяти")
                    vram_fraction = gr.Slider(0.3, 0.98, value=0.80, step=0.01, label="Лимит VRAM (доля)")
                    configure_btn = gr.Button("Применить конфигурацию")
                    config_status = gr.Markdown("Готово к настройке")
            
            # Right panel
            with gr.Column(scale=7, min_width=420, elem_classes=["angel-card"]):
                right_panel = gr.Group()
                with right_panel:
                    flux_components = model_panel_flux()
                    wan_components = model_panel_wan22()
        
        # Logic to toggle panels
        def toggle_panels(selected):
            updates = []
            for c in flux_components:
                updates.append(gr.update(visible=(selected == "Flux")))
            for c in wan_components:
                updates.append(gr.update(visible=(selected == "Wan 2.2")))
            return updates
        
        # Ensure initial visibility
        for c in flux_components:
            c.visible = True
        for c in wan_components:
            c.visible = False
        
        # Bind visibility updates
        model_select.change(
            fn=toggle_panels,
            inputs=[model_select],
            outputs=flux_components + wan_components,
        )
        
        # Memory buttons
        def format_memory_status():
            mi = get_memory_info()
            # Check for psutil warning
            if 'psutil_warning' in mi:
                return mi['psutil_warning']
            if mi.get('gpu_available'):
                return (f"GPU: {mi.get('gpu_name','')} | VRAM: {mi.get('vram_allocated',0):.1f}/{mi.get('vram_total',0):.1f} GiB | "
                        f"RAM: {mi.get('ram_available',0):.1f}/{mi.get('ram_total',0):.1f} GiB")
            else:
                return (f"GPU: N/A | RAM: {mi.get('ram_available',0):.1f}/{mi.get('ram_total',0):.1f} GiB")
        
        refresh_btn = gr.Button("Обновить память")
        memory_status = gr.Markdown(value="—")
        
        configure_btn.click(
            fn=lambda mode, frac: configure_memory_mode(mode, frac),
            inputs=[memory_mode, vram_fraction],
            outputs=[config_status]
        )
        refresh_btn.click(fn=format_memory_status, outputs=[memory_status])
        demo.load(fn=format_memory_status, outputs=[memory_status])
    
    return demo

if __name__ == "__main__":
    interface = create_local_interface()
    interface.launch()
