"""
Universal Memory Manager for PyTorch models (GPU/CPU/Hybrid) with optional Accelerate offloading.

Usage:
    from local.memory_manager import MemoryManager, select_device, limit_vram,
        offload_with_accelerate, offload_cpu_decorator, sequential_frame_processor

Key capabilities:
- Device selection (cpu, cuda, hybrid-auto)
- Dynamic VRAM limit via torch.cuda.set_per_process_memory_fraction
- Manual memory utilities: empty_cache, synchronize, memory stats
- Accelerate integration for automatic device_map and CPU offload
- Sequential processing helpers to minimize peak VRAM
- Pure functions and a class wrapper to fit any model without code duplication

This module is framework-agnostic for model types: pass any torch.nn.Module.
"""
from __future__ import annotations

import os
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import torch

# Optional imports (accelerate may be absent)
try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from accelerate import cpu_offload as accelerate_cpu_offload
except Exception:  # pragma: no cover - optional dependency
    init_empty_weights = None
    load_checkpoint_and_dispatch = None
    accelerate_cpu_offload = None


# -----------------------------
# Device selection and limits
# -----------------------------

def select_device(prefer: str = "auto") -> torch.device:
    """Select a torch.device according to preference.

    prefer: "cpu" | "cuda" | "auto"
      - auto: cuda if available else cpu
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_vram(fraction: float, device_index: int = 0) -> None:
    """Set per-process VRAM fraction (0< fraction <=1) to prevent OOM spikes.
    Safe no-op on CPU-only environments.
    """
    if torch.cuda.is_available():
        fraction = max(0.05, min(1.0, float(fraction)))
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def cuda_sync_empty_cache() -> None:
    """Synchronize and empty CUDA cache (safe no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def memory_stats(device_index: int = 0) -> Dict[str, float]:
    """Return memory stats in GB for RAM/VRAM when available."""
    stats: Dict[str, float] = {}
    # CPU stats via psutil if present
    try:
        import psutil
        vm = psutil.virtual_memory()
        stats.update(
            ram_total_gb=vm.total / (1024 ** 3),
            ram_available_gb=vm.available / (1024 ** 3),
        )
    except Exception:
        pass

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        stats.update(
            vram_total_gb=total,
            vram_allocated_gb=allocated,
            vram_reserved_gb=reserved,
            vram_free_gb=max(0.0, total - allocated),
            gpu_name=torch.cuda.get_device_name(device_index),
        )
    return stats


# -----------------------------
# Accelerate-based offloading
# -----------------------------

def offload_with_accelerate(
    model_ctor: Callable[[], torch.nn.Module],
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    device_map: Union[str, Dict[str, Union[str, int, List[int]]]] = "auto",
    offload_folder: Optional[str] = "offload",
    offload_state_dict: bool = True,
    max_memory: Optional[Dict[Union[int, str], str]] = None,
) -> torch.nn.Module:
    """Load a model with Accelerate device_map and optional offloading.

    model_ctor: zero-arg callable returning an uninitialized nn.Module structure
    checkpoint: path to weights (state_dict or HF checkpoint)
    device_map: e.g., "auto", {0: "3GB", "cpu": "16GB"}, etc.
    offload_folder: destination for offloaded weights/tensors
    max_memory: per-device memory limits like {0: "3GB", "cpu": "16GB"}

    Returns a ready model dispatched across GPU/CPU.
    """
    if init_empty_weights is None or load_checkpoint_and_dispatch is None:
        raise RuntimeError("accelerate is not installed. pip install accelerate")

    os.makedirs(offload_folder or "offload", exist_ok=True)

    with init_empty_weights():
        model = model_ctor()

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint,
        device_map=device_map,
        offload_folder=offload_folder,
        offload_state_dict=offload_state_dict,
        max_memory=max_memory,
    )
    return model


def offload_cpu_decorator(model: torch.nn.Module, execution_device: str = "cuda:0"):
    """Return a decorator that offloads model layers to CPU when idle.
    Requires accelerate; falls back to identity decorator if unavailable.
    """
    if accelerate_cpu_offload is None:
        def identity(fn):
            return fn
        return identity
    return accelerate_cpu_offload(model, execution_device=execution_device)


# -----------------------------
# Sequential processing helper
# -----------------------------

def sequential_frame_processor(
    model: torch.nn.Module,
    frames: Iterable[torch.Tensor],
    device: Union[str, torch.device] = "cuda",
    sync_every: int = 10,
) -> torch.Tensor:
    """Process frames one-by-one to minimize peak VRAM usage.

    Moves each frame to device, computes result, moves back to CPU, clears cache.
    Returns concatenated CPU tensor of results.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    results: List[torch.Tensor] = []
    for i, frame in enumerate(frames):
        frame = frame.to(device)
        with torch.no_grad():
            out = model(frame)
        results.append(out.detach().cpu())
        del frame, out
        cuda_sync_empty_cache()
        if sync_every and (i + 1) % sync_every == 0:
            print(f"Processed {i+1} frames")
    return torch.cat(results, dim=0) if results else torch.empty(0)


# -----------------------------
# High-level class wrapper
# -----------------------------

@dataclass
class MemoryConfig:
    mode: str = "auto"  # "cpu", "cuda", "auto"
    vram_fraction: Optional[float] = None  # e.g., 0.7 for 70%
    device_index: int = 0
    use_accelerate: bool = False
    offload_folder: Optional[str] = "offload"
    device_map: Union[str, Dict[str, Any]] = "auto"
    max_memory: Optional[Dict[Union[int, str], str]] = None


class MemoryManager:
    """Universal memory manager for local AI models.

    Example:
        mm = MemoryManager(MemoryConfig(mode="auto", vram_fraction=0.8))
        device = mm.device
        model = MyModel().to(device)

        # Or, load with accelerate
        if mm.cfg.use_accelerate:
            model = mm.load_with_accelerate(lambda: MyModel(), checkpoint="path")
    """

    def __init__(self, cfg: Optional[MemoryConfig] = None) -> None:
        self.cfg = cfg or MemoryConfig()
        self._device = select_device(self.cfg.mode)
        if self.cfg.vram_fraction is not None:
            limit_vram(self.cfg.vram_fraction, self.cfg.device_index)

    # Properties
    @property
    def device(self) -> torch.device:
        return self._device

    # Utilities
    def sync_and_clear(self) -> None:
        cuda_sync_empty_cache()

    def stats(self) -> Dict[str, float]:
        return memory_stats(self.cfg.device_index)

    # Accelerate integration
    def load_with_accelerate(self, model_ctor: Callable[[], torch.nn.Module], checkpoint: Optional[Union[str, os.PathLike]] = None) -> torch.nn.Module:
        if not self.cfg.use_accelerate:
            raise RuntimeError("Accelerate not enabled in MemoryConfig(use_accelerate=True)")
        return offload_with_accelerate(
            model_ctor,
            checkpoint=checkpoint,
            device_map=self.cfg.device_map,
            offload_folder=self.cfg.offload_folder,
            max_memory=self.cfg.max_memory,
        )

    # Sequential helper passthrough
    def process_frames(self, model: torch.nn.Module, frames: Iterable[torch.Tensor], sync_every: int = 10) -> torch.Tensor:
        return sequential_frame_processor(model, frames, device=self.device, sync_every=sync_every)


# -----------------------------
# Minimal integration notes for UI (local_interface.py)
# -----------------------------
# The UI should import MemoryManager and MemoryConfig, construct a manager once,
# and use manager.device for all tensors/models, avoiding per-model duplication.
# Example sketch in local_interface.py:
#   from local.memory_manager import MemoryManager, MemoryConfig
#   mm = MemoryManager(MemoryConfig(mode="auto", vram_fraction=0.75, use_accelerate=False))
#   device = mm.device
#   model = MyModel().to(device)
#   ... use mm.stats() for live memory, mm.sync_and_clear() on demand ...
