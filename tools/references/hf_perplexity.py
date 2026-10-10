"""Perplexity with HF transformers over the exact windows llm-ppl uses.

Usage:
    python hf_perplexity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --ids model/wikitext2_test.ids --windows 40
"""

import argparse
import math
import time

import numpy as np

BOS_ID = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF repo id or local dir")
    parser.add_argument("--ids", required=True, help="int32 ids from tokenize_text.py")
    parser.add_argument("--ctx", type=int, default=512, help="window length, BOS included")
    parser.add_argument("--windows", type=int, default=0, help="first N windows only (default: all)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    text = np.fromfile(args.ids, dtype="<i4")
    stride = args.ctx - 1
    windows = len(text) // stride
    if args.windows > 0:
        windows = min(windows, args.windows)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.float().to(args.device).eval()

    total_nll = 0.0
    scored = 0
    start = time.time()
    with torch.inference_mode():
        for w in range(windows):
            window = np.concatenate([[BOS_ID], text[w * stride:(w + 1) * stride]]).astype(np.int64)
            ids = torch.from_numpy(window).to(args.device)[None]
            logits = model(ids).logits[0].double()
            logp = torch.log_softmax(logits[:-1], dim=-1)
            total_nll -= logp.gather(1, ids[0, 1:, None]).sum().item()
            scored += stride
            if (w + 1) % 10 == 0 or w + 1 == windows:
                print(f"  window {w + 1:4d}/{windows}  running ppl {math.exp(total_nll / scored):.4f}", flush=True)

    mean_nll = total_nll / scored
    print(f"tokens scored {scored}\nmean nll      {mean_nll:.6f}\nperplexity    {math.exp(mean_nll):.4f}\n"
          f"time          {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
