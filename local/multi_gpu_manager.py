#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU Manager for WAN 2.5 Local Mode
Provides universal multi-GPU support for local AI models

Author: The Angel Studio
Created: October 4, 2025
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU setup."""
    strategy: str = 'data_parallel'  # 'data_parallel', 'ddp', 'model_parallel'
    device_ids: Optional[List[int]] = None
    output_device: Optional[int] = None
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
class MultiGPUManager:
    """
    Universal Multi-GPU Manager for Local AI Models
    
    Supports:
    - DataParallel (DP) for simple inference/training
    - DistributedDataParallel (DDP) for advanced multi-node/multi-GPU
    - Model Parallel for very large models
    - Automatic GPU detection and configuration
    """
    
    def __init__(self, config: Optional[MultiGPUConfig] = None):
        self.config = config or MultiGPUConfig()
        self.device_count = torch.cuda.device_count()
        self.available_devices = list(range(self.device_count))
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        
        # Initialize device info
        self._init_device_info()
        
    def _init_device_info(self):
        """Initialize GPU device information."""
        self.device_info = []
        
        if self.device_count == 0:
            logger.warning("No CUDA devices available. Falling back to CPU.")
            return
            
        logger.info(f"Found {self.device_count} CUDA device(s):")
        
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}"
            }
            self.device_info.append(device_info)
            logger.info(f"  Device {i}: {device_info['name']} ({device_info['total_memory_gb']:.1f} GB)")
    
    def check_multi_gpu_requirements(self) -> Tuple[bool, str]:
        """Check if multi-GPU setup is possible and beneficial."""
        if self.device_count < 2:
            return False, f"Only {self.device_count} GPU(s) available. Multi-GPU requires 2+ GPUs."
        
        # Check memory consistency
        memories = [info['total_memory_gb'] for info in self.device_info]
        if max(memories) - min(memories) > 2.0:  # More than 2GB difference
            logger.warning("GPUs have significantly different memory sizes. This may cause imbalances.")
        
        # Check compute capability
        capabilities = [info['compute_capability'] for info in self.device_info]
        if len(set(capabilities)) > 1:
            logger.warning("GPUs have different compute capabilities. Performance may vary.")
        
        return True, f"Multi-GPU ready: {self.device_count} devices available"
    
    def setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup DataParallel (DP) for simple multi-GPU inference/training."""
        if self.device_count < 2:
            logger.warning("DataParallel requires multiple GPUs. Using single GPU.")
            return model.cuda()
        
        # Configure device IDs
        device_ids = self.config.device_ids or self.available_devices
        output_device = self.config.output_device or device_ids[0]
        
        logger.info(f"Setting up DataParallel on devices: {device_ids}")
        logger.info(f"Output device: {output_device}")
        
        # Move model to GPU and wrap with DataParallel
        model = model.cuda(device_ids[0])
        model = DP(model, device_ids=device_ids, output_device=output_device)
        
        return model
    
    def setup_distributed_data_parallel(self, model: nn.Module, rank: int = 0, world_size: int = None) -> nn.Module:
        """Setup DistributedDataParallel (DDP) for advanced multi-GPU training."""
        if world_size is None:
            world_size = self.device_count
        
        self.local_rank = rank
        self.world_size = world_size
        self.is_distributed = True
        
        # Initialize process group
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size
            )
        
        # Set device
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        logger.info(f"Setting up DDP: rank {rank}/{world_size} on device {device}")
        
        # Move model to GPU and wrap with DDP
        model = model.to(device)
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=self.config.find_unused_parameters,
            bucket_cap_mb=self.config.bucket_cap_mb,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph
        )
        
        return model
    
    def setup_model_parallel(self, model: nn.Module, split_points: List[str]) -> nn.Module:
        """Setup Model Parallel for very large models that don't fit on single GPU."""
        logger.info("Setting up Model Parallel (experimental)")
        
        # This is a basic implementation - actual model parallel setup
        # depends heavily on the specific model architecture
        devices = self.available_devices[:len(split_points)+1]
        
        # Move different parts of the model to different GPUs
        # This is a simplified example - real implementation would need
        # model-specific splitting logic
        
        return model
    
    def launch_multi_gpu(self, model: nn.Module, data: Any = None, strategy: str = None) -> Union[nn.Module, Tuple[nn.Module, Any]]:
        """
        Universal multi-GPU launcher.
        
        Args:
            model: PyTorch model to parallelize
            data: Input data (optional, for data parallel strategies)
            strategy: 'data_parallel', 'ddp', or 'model_parallel'
        
        Returns:
            Parallelized model or tuple of (model, processed_data)
        """
        strategy = strategy or self.config.strategy
        
        # Check multi-GPU requirements
        is_ready, message = self.check_multi_gpu_requirements()
        logger.info(message)
        
        if not is_ready:
            logger.info("Falling back to single GPU/CPU")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            return model.to(device)
        
        try:
            if strategy == 'data_parallel' or strategy == 'dp':
                model = self.setup_data_parallel(model)
                
            elif strategy == 'ddp' or strategy == 'distributed':
                # For DDP, we typically launch multiple processes
                # This is a single-process version for testing
                model = self.setup_distributed_data_parallel(model, rank=0)
                
            elif strategy == 'model_parallel' or strategy == 'mp':
                # Model parallel requires knowing where to split the model
                # This is a placeholder - real implementation needs model-specific logic
                logger.warning("Model Parallel not fully implemented. Using DataParallel.")
                model = self.setup_data_parallel(model)
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Log memory usage after setup
            self._log_memory_usage()
            
            return model
            
        except Exception as e:
            logger.error(f"Multi-GPU setup failed: {e}")
            logger.info("Falling back to single GPU")
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            return model.to(device)
    
    def _log_memory_usage(self):
        """Log current GPU memory usage."""
        for i in range(self.device_count):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
    
    def cleanup(self):
        """Cleanup distributed resources."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")

# Convenience functions
def launch_multi_gpu(model: nn.Module, data: Any = None, strategy: str = 'data_parallel', **kwargs) -> nn.Module:
    """
    Convenience function to launch multi-GPU processing.
    
    Args:
        model: PyTorch model to parallelize
        data: Input data (optional)
        strategy: 'data_parallel', 'ddp', or 'model_parallel'
        **kwargs: Additional configuration options
    
    Returns:
        Parallelized model
    
    Example:
        >>> model = MyModel()
        >>> multi_gpu_model = launch_multi_gpu(model, strategy='data_parallel')
        >>> output = multi_gpu_model(input_data)
    """
    config = MultiGPUConfig(strategy=strategy, **kwargs)
    manager = MultiGPUManager(config)
    return manager.launch_multi_gpu(model, data, strategy)

def get_gpu_info() -> List[Dict[str, Any]]:
    """Get information about available GPUs."""
    manager = MultiGPUManager()
    return manager.device_info

def check_multi_gpu_ready() -> Tuple[bool, str]:
    """Check if system is ready for multi-GPU processing."""
    manager = MultiGPUManager()
    return manager.check_multi_gpu_requirements()

# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic usage
    print("=== Multi-GPU Manager Test ===")
    
    # Check system readiness
    is_ready, message = check_multi_gpu_ready()
    print(f"Multi-GPU Status: {message}")
    
    # Show GPU info
    gpu_info = get_gpu_info()
    for info in gpu_info:
        print(f"GPU {info['id']}: {info['name']} ({info['total_memory_gb']:.1f} GB)")
    
    # Example 2: Simple model setup
    try:
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(1000, 512)
                self.linear2 = nn.Linear(512, 256)
                self.linear3 = nn.Linear(256, 10)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.relu(self.linear2(x))
                return self.linear3(x)
        
        # Test different strategies
        model = TestModel()
        print("\n=== Testing DataParallel ===")
        
        # Launch multi-GPU with DataParallel
        multi_gpu_model = launch_multi_gpu(model, strategy='data_parallel')
        print(f"Model setup complete. Type: {type(multi_gpu_model)}")
        
        # Test inference
        if torch.cuda.is_available():
            test_input = torch.randn(32, 1000).cuda()
            with torch.no_grad():
                output = multi_gpu_model(test_input)
            print(f"Inference test successful. Output shape: {output.shape}")
        
        print("\n=== Multi-GPU Manager Test Complete ===")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
