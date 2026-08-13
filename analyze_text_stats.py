#!/usr/bin/env python3
"""Text mechanism analysis for Raw / Fixed / Multi pair_control corpora.

Follows FOLLOWUP_EXPERIMENT_PLAN.md section 8:
  - char / word / CLIP-token distributions
  - fraction over the 77-token limit
  - lexical diversity (type-token ratio)
  - Add/Remove/Change template-word frequencies
  - empty / malformed output rates
  - NP counts (from the NP-annotated spans)
Run on CPU; safe to run while GPU training is in progress.
"""
import json
import re
import sys
from collections import Counter

from transformers import CLIPTokenizer

FILES = {
    "RAW": "/home/zihali/final_thesis/data/pair_control/train.jsonl",
    "FIXED": "/home/zihali/final_thesis/data/pair_control/train_fixed.jsonl",
    "MULTI": "/home/zihali/final_thesis/data/pair_control/train_multi.jsonl",
}

ADD_REMOVE_WORDS = ["add", "remove", "change", "replace", "swap", "delete", "insert",
                    "convert", "turn", "make", "exclude", "include"]

def tokenize_text(text):
    return re.findall(r"\b[\w'-]+\b", text.lower())

def load(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))
    return rows

def analyze(name, rows, tokenizer):
    texts = [r.get("modification") or r.get("merdcir_modification") or "" for r in rows]
    lens_chars = [len(t) for t in texts]
    lens_words = [len(tokenize_text(t)) for t in texts]
    n_tokens_list = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    over77 = sum(1 for n in n_tokens_list if n > 77) / len(texts)

    # type-token ratio (per-doc mean and corpus-level)
    ttrs = []
    corpus_tokens = []
    for t in texts:
        toks = tokenize_text(t)
        if toks:
            ttrs.append(len(set(toks)) / len(toks))
        corpus_tokens.extend(toks)
    corpus_ttr = len(set(corpus_tokens)) / max(1, len(corpus_tokens))

    # template word frequency (normalized per 1000 words)
    word_freq = Counter(corpus_tokens)
    total_words = len(corpus_tokens)
    tmpl = {w: word_freq.get(w, 0) for w in ADD_REMOVE_WORDS}
    tmpl_per1k = {w: (c / max(1, total_words)) * 1000 for w, c in tmpl.items()}

    # empty / malformed / nps
    empty = sum(1 for t in texts if not t.strip())
    nps_counts = [len(r.get("nps") or []) for r in rows]
    zero_nps = sum(1 for c in nps_counts if c == 0)

    print(f"\n===== {name} (n={len(rows)}) =====")
    print(f"  char length:      mean={sum(lens_chars)/len(lens_chars):.1f}  "
          f"median={sorted(lens_chars)[len(lens_chars)//2]}  max={max(lens_chars)}")
    print(f"  word count:       mean={sum(lens_words)/len(lens_words):.1f}  "
          f"median={sorted(lens_words)[len(lens_words)//2]}  max={max(lens_words)}")
    print(f"  CLIP tokens:      mean={sum(n_tokens_list)/len(n_tokens_list):.2f}  "
          f"median={sorted(n_tokens_list)[len(n_tokens_list)//2]}  max={max(n_tokens_list)}")
    print(f"  over 77 tokens:   {over77*100:.2f}%")
    print(f"  TTR (doc-mean):   {sum(ttrs)/max(1,len(ttrs)):.4f}   TTR (corpus): {corpus_ttr:.4f}")
    print(f"  empty texts:      {empty} ({empty/len(texts)*100:.2f}%)")
    print(f"  NPs/doc:          mean={sum(nps_counts)/len(nps_counts):.2f}  zero-NP={zero_nps} "
          f"({zero_nps/len(rows)*100:.2f}%)")
    print(f"  template words/1k:")
    for w, c in tmpl_per1k.items():
        print(f"    {w:>10s}: {c:7.2f}  (raw count {tmpl[w]})")
    return {
        "n": len(rows),
        "chars_mean": sum(lens_chars) / len(lens_chars),
        "words_mean": sum(lens_words) / len(lens_words),
        "tokens_mean": sum(n_tokens_list) / len(n_tokens_list),
        "tokens_max": max(n_tokens_list),
        "over77": over77,
        "ttr_doc_mean": sum(ttrs) / max(1, len(ttrs)),
        "ttr_corpus": corpus_ttr,
        "empty_rate": empty / len(texts),
        "nps_mean": sum(nps_counts) / len(nps_counts),
        "zero_nps_rate": zero_nps / len(rows),
        "template_per1k": tmpl_per1k,
    }

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    results = {}
    for name, path in FILES.items():
        rows = load(path, limit)
        results[name] = analyze(name, rows, tokenizer)
    out = "/home/zihali/final_thesis/checkpoints/pair_control_text_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
