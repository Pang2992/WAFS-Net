"""
Model definitions for WAFS-Net.
"""

from .wafs_net import (
    WAFSNet,
    DualWaveletBinaryModel,
    build_wafs_net,
)

__all__ = [
    "WAFSNet",
    "DualWaveletBinaryModel",
    "build_wafs_net",
]
