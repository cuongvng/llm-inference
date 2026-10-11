"""Group-wise weight quantization: the definition the CUDA kernels must reproduce.

A weight matrix is [N, K] row-major. Each row is cut along K into groups of G
consecutive values, and every group gets its own FP16 scale (and, for INT4, an
FP16 min), stored as [N, K/G].

  INT8, symmetric:   w ~= q * s          q in [-127, 127], s = max|w| / 127
  INT4, asymmetric:  w ~= q * s + m      q in [0, 15],     s = (max - min) / 15, m = min
                     packed 2 per byte along K, even k in the low nibble.

"""

from __future__ import annotations

import numpy as np


def _groups(w: np.ndarray, group: int) -> np.ndarray:
    n, k = w.shape
    if group % 32 != 0 or k % group != 0:
        raise ValueError(f"group {group} must be a multiple of 32 and divide K={k}")
    return w.astype(np.float32).reshape(n, k // group, group)


def _expand(per_group: np.ndarray, group: int) -> np.ndarray:
    return np.repeat(per_group.astype(np.float32), group, axis=1)


def quantize_int8(w: np.ndarray, group: int):
    """[N, K] float -> (q int8 [N, K], scales fp16 [N, K/G])."""
    g = _groups(w, group)
    scales = (np.abs(g).max(axis=2) / 127.0).astype(np.float16)
    s = scales.astype(np.float32)[..., None]
    q = np.rint(np.divide(g, s, out=np.zeros_like(g), where=s > 0))
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.reshape(w.shape), scales


def dequantize_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    group = q.shape[1] // scales.shape[1]
    return q.astype(np.float32) * _expand(scales, group)


def quantize_int4(w: np.ndarray, group: int):
    """[N, K] float -> (packed uint8 [N, K/2], scales fp16 [N, K/G], mins fp16 [N, K/G])."""
    g = _groups(w, group)
    mins = g.min(axis=2).astype(np.float16)
    scales = ((g.max(axis=2) - mins.astype(np.float32)) / 15.0).astype(np.float16)
    s = scales.astype(np.float32)[..., None]
    m = mins.astype(np.float32)[..., None]
    q = np.rint(np.divide(g - m, s, out=np.zeros_like(g), where=s > 0))
    q = np.clip(q, 0, 15).astype(np.uint8).reshape(w.shape)
    return pack_int4(q), scales, mins


def dequantize_int4(packed: np.ndarray, scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    q = unpack_int4(packed).astype(np.float32)
    group = q.shape[1] // scales.shape[1]
    return q * _expand(scales, group) + _expand(mins, group)


def pack_int4(q: np.ndarray) -> np.ndarray:
    """[N, K] values in [0, 15] -> [N, K/2] bytes, even k in the low nibble."""
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    q = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    return q
