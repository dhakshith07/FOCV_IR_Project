"""
utils/system_info.py
Collects local system information for the dashboard status panel.
"""

import platform
import sys
import psutil
from pathlib import Path
from typing import Dict, Any


def get_system_info() -> Dict[str, Any]:
    """Return a dictionary of system diagnostics."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home()))

    return {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "python_version": sys.version.split(" ")[0],
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": round(mem.total / (1024 ** 3), 1),
        "ram_used_gb": round(mem.used / (1024 ** 3), 1),
        "ram_percent": mem.percent,
        "disk_total_gb": round(disk.total / (1024 ** 3), 1),
        "disk_used_gb": round(disk.used / (1024 ** 3), 1),
        "disk_percent": disk.percent,
    }


def check_gpu_available() -> bool:
    """Check if a CUDA-capable GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_torch_info() -> Dict[str, Any]:
    """Return PyTorch version and device info."""
    try:
        import torch
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    except ImportError:
        return {
            "torch_version": "Not installed",
            "cuda_available": False,
            "device": "N/A",
        }
