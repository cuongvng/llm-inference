"""Writer for the custom `.llmbin` format documented in docs/binary_format.md.

Stdlib-only (struct) so it has no dependency on numpy/torch — those are only
needed by convert_weights.py to *read* HuggingFace checkpoints, not to write
this format. Keep this file and docs/binary_format.md in sync by hand; the
C++ mirror is engine/src/loader.cpp + engine/include/llm/config.h.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"LLM1"
VERSION = 1

DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_INT8 = 2
DTYPE_INT4 = 3

_DTYPE_NAMES = {DTYPE_FP32: "fp32", DTYPE_FP16: "fp16", DTYPE_INT8: "int8", DTYPE_INT4: "int4"}


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float

    def pack(self) -> bytes:
        return struct.pack(
            "<8I2f",
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.intermediate_size,
            self.max_seq_len,
            self.rope_theta,
            self.rms_norm_eps,
        )


@dataclass
class Tensor:
    name: str
    dtype: int
    shape: tuple
    data: bytes

    def __post_init__(self):
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(f"unknown dtype {self.dtype}")
        if len(self.data) == 0:
            raise ValueError(f"tensor {self.name!r} has empty data")


@dataclass
class ModelWriter:
    config: ModelConfig
    tensors: list = field(default_factory=list)

    def add_tensor(self, name: str, dtype: int, shape, data: bytes) -> None:
        self.tensors.append(Tensor(name=name, dtype=dtype, shape=tuple(shape), data=data))

    def write(self, path: str) -> None:
        directory = b""
        data_section = b""
        offset = 0
        for t in self.tensors:
            name_bytes = t.name.encode("utf-8")
            entry = struct.pack("<I", len(name_bytes)) + name_bytes
            entry += struct.pack("<II", t.dtype, len(t.shape))
            entry += struct.pack(f"<{len(t.shape)}I", *t.shape)
            entry += struct.pack("<QQ", offset, len(t.data))
            directory += entry
            data_section += t.data
            offset += len(t.data)

        header = MAGIC + struct.pack("<I", VERSION)
        header += self.config.pack()
        header += struct.pack("<I", len(self.tensors))

        with open(path, "wb") as f:
            f.write(header)
            f.write(directory)
            f.write(data_section)


def pack_fp16_row(values) -> bytes:
    """Pack an iterable of floats as consecutive IEEE-754 half-precision values."""
    return b"".join(struct.pack("<e", v) for v in values)
