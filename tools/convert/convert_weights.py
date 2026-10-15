"""HuggingFace checkpoint -> .llmbin.
Targets Llama-architecture checkpoints (TinyLlama-1.1B and compatible): RoPE,
RMSNorm, GQA, SwiGLU.

Usage:
    python convert_weights.py --model <hf-repo-or-local-dir> --out model.llmbin \\
        [--quant fp16|int8|int4] [--group 128]
"""

import argparse
import sys

import binformat
import quantize


def _load_hf_model(model_path: str):
    import torch  # noqa: F401  (import lazily -- keeps binformat.py dependency-free)
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float16")
    model.eval()
    return config, model


def _fp16_bytes(tensor) -> bytes:
    return tensor.detach().to("cpu", dtype=__import__("torch").float16).contiguous().numpy().tobytes()


def convert(model_path: str, out_path: str, quant: str = "fp16", group: int = 128) -> None:
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

    def add_linear(name: str, hf_key: str):
        if quant == "fp16":
            return add(name, hf_key)
        w = sd[hf_key].detach().to("cpu", dtype=__import__("torch").float16).float().numpy()
        base = name.removesuffix(".weight")
        if quant == "int8":
            q, scales = quantize.quantize_int8(w, group)
            writer.add_tensor(name, binformat.DTYPE_INT8, w.shape, q.tobytes())
        else:
            q, scales, mins = quantize.quantize_int4(w, group)
            writer.add_tensor(name, binformat.DTYPE_INT4, w.shape, q.tobytes())
            writer.add_tensor(base + ".mins", binformat.DTYPE_FP16, mins.shape, mins.tobytes())
        writer.add_tensor(base + ".scales", binformat.DTYPE_FP16, scales.shape, scales.tobytes())

    add("tok_embeddings.weight", "model.embed_tokens.weight")
    add("output_norm.weight", "model.norm.weight")
    add_linear("lm_head.weight", "lm_head.weight")

    for layer in range(config.num_layers):
        p = f"model.layers.{layer}."
        o = f"layers.{layer}."
        add(o + "attn_norm.weight", p + "input_layernorm.weight")
        add_linear(o + "attn.wq.weight", p + "self_attn.q_proj.weight")
        add_linear(o + "attn.wk.weight", p + "self_attn.k_proj.weight")
        add_linear(o + "attn.wv.weight", p + "self_attn.v_proj.weight")
        add_linear(o + "attn.wo.weight", p + "self_attn.o_proj.weight")
        add(o + "ffn_norm.weight", p + "post_attention_layernorm.weight")
        add_linear(o + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight")
        add_linear(o + "mlp.up_proj.weight", p + "mlp.up_proj.weight")
        add_linear(o + "mlp.down_proj.weight", p + "mlp.down_proj.weight")

    writer.write(out_path)
    total = sum(len(t.data) for t in writer.tensors)
    print(f"wrote {out_path} ({quant}{'' if quant == 'fp16' else f', group {group}'}): "
          f"{len(writer.tensors)} tensors, {total / 2**20:.1f} MiB, vocab={config.vocab_size}, "
          f"layers={config.num_layers}, hidden={config.hidden_size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--out", required=True, help="output .llmbin path")
    ap.add_argument("--quant", choices=["fp16", "int8", "int4"], default="fp16",
                    help="format of the matmul weights (default fp16)")
    ap.add_argument("--group", type=int, default=128, help="values per scale along K (default 128)")
    args = ap.parse_args()
    convert(args.model, args.out, args.quant, args.group)


if __name__ == "__main__":
    sys.exit(main())
