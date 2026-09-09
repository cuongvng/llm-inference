"""Build a tiny synthetic model file for engine/tests/test_loader.cu.

Outputs:
  - <out_dir>/tiny_model.llmbin   the model file, loaded by the C++ test
  - <out_dir>/expected_row.bin    raw fp16 bytes of the embedding row for
                                  TOKEN_ID, for a byte-exact memcmp in C++

No numpy/torch: struct's 'e' format code (Python 3.6+) packs IEEE-754 half
directly, which is all a synthetic fixture needs.
"""

import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import binformat  # noqa: E402

VOCAB_SIZE = 17
HIDDEN_SIZE = 8
TOKEN_ID = 5  # arbitrary, but fixed so both sides agree


def embedding_value(token: int, col: int) -> float:
    """Deterministic, human-checkable filler: distinct per (token, col)."""
    return token * 0.1 + col * 0.01


def build_embedding_table_fp16() -> bytes:
    rows = []
    for token in range(VOCAB_SIZE):
        row = [embedding_value(token, col) for col in range(HIDDEN_SIZE)]
        rows.append(binformat.pack_fp16_row(row))
    return b"".join(rows)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(2)
    out_dir = sys.argv[1]

    config = binformat.ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        intermediate_size=16,
        max_seq_len=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )

    writer = binformat.ModelWriter(config=config)
    embedding_bytes = build_embedding_table_fp16()
    writer.add_tensor(
        name="tok_embeddings.weight",
        dtype=binformat.DTYPE_FP16,
        shape=(VOCAB_SIZE, HIDDEN_SIZE),
        data=embedding_bytes,
    )
    writer.write(f"{out_dir}/tiny_model.llmbin")

    expected_row = binformat.pack_fp16_row(
        embedding_value(TOKEN_ID, col) for col in range(HIDDEN_SIZE)
    )
    with open(f"{out_dir}/expected_row.bin", "wb") as f:
        f.write(expected_row)


if __name__ == "__main__":
    main()
