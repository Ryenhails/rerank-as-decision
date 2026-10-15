"""Latency per query on one GPU at concurrency 1: every system reranks the same top-K ViDoRe V3 pages of the same
English queries. Latency is the wall time from request to scores, preprocessing included; page images are encoded
before timing. The first --warmup queries are run and dropped.

    # Qwen3-VL-Reranker with the released code (bf16, FlashAttention-2, the K pairs one after another)
    python scripts/latency.py qwen --vidore-out runs/vidore3 --qwen-repo Qwen3-VL-Embedding \\
        --models 2B=Qwen/Qwen3-VL-Reranker-2B 8B=Qwen/Qwen3-VL-Reranker-8B --out runs/latency
    # GlanceRank against a running server
    python scripts/latency.py vlm --vidore-out runs/vidore3 --backend chat --url http://127.0.0.1:8000 \\
        --model Qwen/Qwen3.8-27B --readout listwise --name base_listwise --out runs/latency

`--vidore-out` holds the first-stage rankings written by `eval_vidore3.py emb`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_vidore3 import QUERY_INSTRUCTION, iter_pages, load_meta, make_ranker, select  # noqa: E402

from glancerank import to_data_url  # noqa: E402


def workload(vidore_out, datasets, n_per_dataset, k):
    """[(dataset, query text, top-k corpus ids)] for the first n English queries of each dataset."""
    items = []
    for d in datasets:
        qs, _ = load_meta(d)
        top = json.load(open(f"{vidore_out}/{d}/emb_top100.json"))
        for q in select(qs, "english")[:n_per_dataset]:
            items.append((d, q["query"], top[str(q["query_id"])][:k]))
    return items


def summary(lat, tok, calls):
    return {"n": len(lat), "mean_s": statistics.mean(lat), "median_s": statistics.median(lat),
            "p90_s": sorted(lat)[int(0.9 * (len(lat) - 1))], "mean_prefill_tokens": statistics.mean(tok),
            "calls_per_query": calls}


def run_qwen(args, items, pages):
    import torch
    sys.path.insert(0, args.qwen_repo)
    from src.models.qwen3_vl_reranker import Qwen3VLReranker
    res = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        rr = Qwen3VLReranker(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        lat, tok = [], []
        for i, (d, qtext, cids) in enumerate(items):
            docs = [{"image": pages[(d, c)]} for c in cids]
            torch.cuda.synchronize()
            t = time.perf_counter()
            rr.process({"instruction": QUERY_INSTRUCTION, "query": {"text": qtext}, "documents": docs})
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            fmt = rr.format_mm_instruction  # token count outside the timed region
            pairs = [fmt(qtext, None, None, None, doc["image"], None, instruction=QUERY_INSTRUCTION) for doc in docs]
            if i >= args.warmup:
                lat.append(dt)
                tok.append(int(rr.tokenize(pairs)["attention_mask"].sum()))
        res[f"Qwen3-VL-Reranker-{name}"] = summary(lat, tok, len(items[0][2]))
        print(name, res[f"Qwen3-VL-Reranker-{name}"], flush=True)
        del rr
        torch.cuda.empty_cache()
    return res


def run_vlm(args, items, pages):
    ranker = make_ranker(args)
    urls = {key: to_data_url(im, args.max_pixels) for key, im in pages.items()}
    lat, tok, calls = [], [], []
    for i, (d, qtext, cids) in enumerate(items):
        t = time.perf_counter()
        r = ranker.rank(qtext, [{"image": urls[(d, c)]} for c in cids], instruction=QUERY_INSTRUCTION)
        dt = time.perf_counter() - t
        if i >= args.warmup:
            lat.append(dt)
            tok.append(r.prompt_tokens)
            calls.append(r.calls)
    res = {args.name: summary(lat, tok, statistics.mean(calls))}
    print(args.name, res[args.name], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["qwen", "vlm"])
    ap.add_argument("--vidore-out", required=True, help="the --out of eval_vidore3.py (first-stage rankings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["computer_science", "hr"])
    ap.add_argument("--n", type=int, default=55, help="English queries per dataset (first --warmup overall dropped)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
    ap.add_argument("--qwen-repo", help="qwen: a checkout of github.com/QwenLM/Qwen3-VL-Embedding")
    ap.add_argument("--models", nargs="*", default=[], help="qwen: NAME=model name or path")
    ap.add_argument("--backend", choices=["chat", "jev"], default="chat")
    ap.add_argument("--url", help="vlm: server root (key from $VLLM_API_KEY)")
    ap.add_argument("--model", help="vlm --backend chat: the served model name")
    ap.add_argument("--tokenizer", help="vlm --backend chat: tokenizer name or path (default: --model)")
    ap.add_argument("--readout", choices=["listwise", "pointwise"], default="listwise")
    ap.add_argument("--name", default="glancerank", help="vlm: the row name in the output")
    args = ap.parse_args()
    items = workload(args.vidore_out, args.datasets, args.n, args.topk)
    pages = {}
    for d in sorted({it[0] for it in items}):
        for cid, im in iter_pages(d, {c for it in items if it[0] == d for c in it[2]}):
            pages[(d, cid)] = im
    res = run_qwen(args, items, pages) if args.stage == "qwen" else run_vlm(args, items, pages)
    os.makedirs(args.out, exist_ok=True)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    meta = {"gpu": smi.stdout.strip(), "datasets": args.datasets, "queries": len(items), "warmup_dropped": args.warmup,
            "topk": args.topk, "max_pixels": args.max_pixels, "concurrency": 1}
    path = f"{args.out}/latency_{args.stage if args.stage == 'qwen' else args.name}.json"
    json.dump({"meta": meta, "results": res}, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
