"""Plain text file -> flat little-endian int32 token ids (no header), for llm-ppl.

Usage:
    python tokenize_text.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --text model/wikitext2_test.raw --out model/wikitext2_test.ids
"""

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF repo id or local dir whose tokenizer to use")
    parser.add_argument("--text", required=True, help="UTF-8 text file, e.g. WikiText-2 wiki.test.raw")
    parser.add_argument("--out", required=True, help="output path for the int32 ids")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.text, encoding="utf-8") as f:
        text = f.read()

    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype="<i4")
    ids.tofile(args.out)
    print(f"wrote {args.out}: {ids.size} tokens from {len(text)} characters")


if __name__ == "__main__":
    main()
