"""HuggingFace checkpoint -> .llmbin (docs/binary_format.md), FP16, unquantized.

Dev-time only -- needs torch/transformers/safetensors (tools/convert/requirements.txt),
none of which are runtime dependencies of the C++ engine (DESIGN_QA.md). Quantization
(INT8/INT4, group-wise) is layered on top of this in quantize.py at M4; this script
always emits FP16 so its output can serve as the M2/M3 torch-reference baseline too.

Targets Llama-architecture checkpoints (TinyLlama-1.1B and compatible): RoPE,
RMSNorm, GQA, SwiGLU. Tensor names in the output file match what
engine/src/model.cpp expects to find via ModelFile::find().

Usage:
    python convert_weights.py --model <hf-repo-or-local-dir> --out model.llmbin
"""

import argparse
import sys

import binformat


def _load_hf_model(model_path: str):
    import torch  # noqa: F401  (import lazily -- keeps binformat.py dependency-free)
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float16")
    model.eval()
    return config, model


def _fp16_bytes(tensor) -> bytes:
    return tensor.detach().to("cpu", dtype=__import__("torch").float16).contiguous().numpy().tobytes()


def convert(model_path: str, out_path: str) -> None:
    hf_config, model = _load_hf_model(model_path)
    sd = model.state_dict()

    config = binformat.ModelConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 2048),
        rope_theta=float(getattr(hf_config, "rope_theta", 10000.0)),
        rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-5)),
    )

    writer = binformat.ModelWriter(config=config)

    def add(name: str, hf_key: str):
        t = sd[hf_key]
        writer.add_tensor(name, binformat.DTYPE_FP16, tuple(t.shape), _fp16_bytes(t))

    add("tok_embeddings.weight", "model.embed_tokens.weight")
    add("output_norm.weight", "model.norm.weight")
    add("lm_head.weight", "lm_head.weight")

    for layer in range(config.num_layers):
        p = f"model.layers.{layer}."
        o = f"layers.{layer}."
        add(o + "attn_norm.weight", p + "input_layernorm.weight")
        add(o + "attn.wq.weight", p + "self_attn.q_proj.weight")
        add(o + "attn.wk.weight", p + "self_attn.k_proj.weight")
        add(o + "attn.wv.weight", p + "self_attn.v_proj.weight")
        add(o + "attn.wo.weight", p + "self_attn.o_proj.weight")
        add(o + "ffn_norm.weight", p + "post_attention_layernorm.weight")
        add(o + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight")
        add(o + "mlp.up_proj.weight", p + "mlp.up_proj.weight")
        add(o + "mlp.down_proj.weight", p + "mlp.down_proj.weight")

    writer.write(out_path)
    print(f"wrote {out_path}: {len(writer.tensors)} tensors, vocab={config.vocab_size}, "
          f"layers={config.num_layers}, hidden={config.hidden_size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--out", required=True, help="output .llmbin path")
    args = ap.parse_args()
    convert(args.model, args.out)


if __name__ == "__main__":
    sys.exit(main())
