"""ViDoRe V3 (8 public datasets): Qwen3-VL-Embedding-2B retrieves the top-100 pages, then rerankers refine its top-K.

Protocol (it reproduces the published Qwen3-VL-Embedding-2B score, 52.9 vs 53.0 here): queries of all six languages
are pooled within each dataset, nDCG@10 per dataset, mean over the 8 datasets. `score` also reports English only.

Stages, one per call; results go to --out/<dataset>/:
  emb     Qwen3-VL-Embedding-2B (the Qwen repo's Qwen3VLEmbedder) -> emb_top100.json
  rr      a Qwen3-VL-Reranker with the repo's own per-pair scoring on the top-K -> rr_<name>_top<K>_<langs>.json
  rerank  VLM-as-Jev on the top-K page images -> <name>_top<K>_<langs>.json (+ _calls.jsonl, resumable)
  score   nDCG@10 of every system found -> summary_{pooled,english}.json

    python scripts/eval_vidore3.py emb --out runs/vidore3 --qwen-repo Qwen3-VL-Embedding \
        --model Qwen/Qwen3-VL-Embedding-2B
    python scripts/eval_vidore3.py rr --out runs/vidore3 --qwen-repo Qwen3-VL-Embedding \
        --model Qwen/Qwen3-VL-Reranker-2B --name 2B --topk 16 --langs all
    python scripts/eval_vidore3.py rerank --out runs/vidore3 --backend chat --url http://127.0.0.1:8000 \
        --model Qwen/Qwen3.8-27B --name base_listwise --topk 16 --langs all
    python scripts/eval_vidore3.py score  --out runs/vidore3

The datasets are read from the Hugging Face Hub (vidore/vidore_v3_<dataset>) as parquet; pages stay in memory.
"""
import argparse
import concurrent.futures as cf
import glob
import io
import json
import math
import os
import sys
import threading
import time

from vlm_as_jev import MAX_OPTIONS, ChatBackend, JevDecideBackend, Reranker, to_data_url

DATASETS = ["computer_science", "energy", "finance_en", "finance_fr", "hr", "industrial", "pharmaceuticals", "physics"]
QUERY_INSTRUCTION = "Find a document image that matches the given query."  # the MMEB VisDoc query instruction


def snapshot(dataset):
    from huggingface_hub import snapshot_download
    return snapshot_download(f"vidore/vidore_v3_{dataset}", repo_type="dataset")


def load_meta(dataset):
    import pyarrow.parquet as pq
    s = snapshot(dataset)
    qs = pq.read_table(glob.glob(f"{s}/queries/*.parquet"), columns=["query_id", "query", "language"]).to_pylist()
    qrels = {}
    for r in pq.read_table(glob.glob(f"{s}/qrels/*.parquet"), columns=["query_id", "corpus_id", "score"]).to_pylist():
        qrels.setdefault(str(r["query_id"]), {})[str(r["corpus_id"])] = int(r["score"])
    return qs, qrels


def iter_pages(dataset, ids=None):
    """(corpus_id, PIL image) in file order, optionally only `ids`."""
    import pyarrow.parquet as pq
    from PIL import Image
    want = set(ids) if ids is not None else None
    for f in sorted(glob.glob(f"{snapshot(dataset)}/corpus/*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["corpus_id", "image"]):
            for cid, img in zip(b.column("corpus_id").to_pylist(), b.column("image").to_pylist()):
                if want is None or cid in want:
                    yield cid, Image.open(io.BytesIO(img["bytes"])).convert("RGB")


def select(qs, langs):
    return [q for q in qs if langs == "all" or q["language"] == langs]


def ndcg10(rank, gains):
    dcg = sum(gains.get(str(c), 0) / math.log2(i + 2) for i, c in enumerate(rank[:10]))
    ideal = sorted((v for v in gains.values() if v > 0), reverse=True)[:10]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------- stages
def stage_emb(args):
    import torch
    sys.path.insert(0, args.qwen_repo)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    emb = Qwen3VLEmbedder(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    for d in args.datasets:
        out = f"{args.out}/{d}"
        os.makedirs(out, exist_ok=True)
        if os.path.exists(f"{out}/emb_top100.json"):
            print(f"[{d}] emb done earlier", flush=True)
            continue
        t0 = time.time()
        qs, _ = load_meta(d)
        cids, cvec, buf = [], [], []
        for cid, im in iter_pages(d):
            cids.append(cid)
            buf.append({"image": im})
            if len(buf) == args.batch:
                cvec.append(emb.process(buf).float().cpu())
                buf = []
        if buf:
            cvec.append(emb.process(buf).float().cpu())
        corpus = torch.cat(cvec)
        queries = torch.cat([emb.process([{"text": q["query"], "instruction": QUERY_INSTRUCTION} for q in qs[s:s + 64]])
                             .float().cpu() for s in range(0, len(qs), 64)])
        top = (queries @ corpus.T).topk(min(100, len(cids)), dim=1).indices.tolist()
        ranking = {str(q["query_id"]): [cids[i] for i in row] for q, row in zip(qs, top)}
        json.dump(ranking, open(f"{out}/emb_top100.json", "w"))
        print(f"[{d}] emb: {len(cids)} pages, {len(qs)} queries, {time.time() - t0:.0f}s", flush=True)


def stage_rr(args):
    import torch
    sys.path.insert(0, args.qwen_repo)
    from src.models.qwen3_vl_reranker import Qwen3VLReranker
    rr = Qwen3VLReranker(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    for d in args.datasets:
        out = f"{args.out}/{d}"
        path = f"{out}/rr_{args.name}_top{args.topk}_{args.langs}.json"
        if os.path.exists(path):
            print(f"[{d}] rr {args.name} done earlier", flush=True)
            continue
        t0 = time.time()
        qs, _ = load_meta(d)
        top = json.load(open(f"{out}/emb_top100.json"))
        sel = select(qs, args.langs)[: args.limit]
        pages = dict(iter_pages(d, {c for q in sel for c in top[str(q["query_id"])][: args.topk]}))
        res, lat = {}, []
        for q in sel:
            qid = str(q["query_id"])
            ids = top[qid][: args.topk]
            t = time.time()
            scores = rr.process({"instruction": QUERY_INSTRUCTION, "query": {"text": q["query"]},
                                 "documents": [{"image": pages[c]} for c in ids]})
            lat.append(time.time() - t)
            res[qid] = [c for _, c in sorted(zip(scores, ids), key=lambda x: -x[0])]
        if not args.limit:
            json.dump({"ranking": res, "seconds_per_query": sum(lat) / len(lat), "wall": round(time.time() - t0)},
                      open(path, "w"))
        print(f"[{d}] rr {args.name} top{args.topk} {args.langs}: {len(sel)} queries, "
              f"{sum(lat) / len(lat):.2f} s/query", flush=True)


def make_reranker(args):
    if args.backend == "jev":
        backend = JevDecideBackend(args.url)
    else:
        backend = ChatBackend(args.url, args.model, tokenizer=args.tokenizer)
    return Reranker(backend, readout=args.readout, max_pixels=args.max_pixels)


def stage_rerank(args):
    reranker = make_reranker(args)
    for d in args.datasets:
        out = f"{args.out}/{d}"
        path = f"{out}/{args.name}_top{args.topk}_{args.langs}.json"
        if os.path.exists(path):
            print(f"[{d}] {args.name} done earlier", flush=True)
            continue
        t0 = time.time()
        qs, _ = load_meta(d)
        top = json.load(open(f"{out}/emb_top100.json"))
        sel = select(qs, args.langs)[: args.limit]
        need = {c for q in sel for c in top[str(q["query_id"])][: args.topk]}
        urls = {c: to_data_url(im, args.max_pixels) for c, im in iter_pages(d, need)}  # each page encoded once
        calls_path = f"{out}/{args.name}_top{args.topk}_{args.langs}_calls.jsonl"
        done = {}
        if os.path.exists(calls_path):
            for line in open(calls_path):
                r = json.loads(line)
                done[r["qid"]] = r
        lock, fh = threading.Lock(), open(calls_path, "a")

        def one(q):
            qid = str(q["query_id"])
            if qid in done:
                return done[qid]
            ids = top[qid][: args.topk]
            rec = {"qid": qid, "ids": ids, "probs": None, "prompt_tokens": 0, "calls": 0, "seconds": 0.0, "error": None}
            t = time.time()
            for _ in range(3):
                try:
                    r = reranker.rerank(q["query"], [{"image": urls[c]} for c in ids], instruction=QUERY_INSTRUCTION)
                    rec.update(probs=r.scores, prompt_tokens=r.prompt_tokens, calls=r.calls, error=None)
                    break
                except Exception as e:  # noqa: BLE001 - logged per query, the query keeps the first-stage order
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    time.sleep(5)
            rec["seconds"] = round(time.time() - t, 2)
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            return rec

        with cf.ThreadPoolExecutor(args.workers) as ex:
            recs = list(ex.map(one, sel))
        fh.close()
        res = {}
        for r in recs:
            if r["probs"]:
                p = dict(zip(r["ids"], r["probs"]))
                res[r["qid"]] = sorted(r["ids"], key=lambda c: (-p[c], r["ids"].index(c)))
            else:
                res[r["qid"]] = r["ids"]
        failed = sum(1 for r in recs if not r["probs"])
        if not args.limit:
            json.dump({"ranking": res, "failed": failed, "wall": round(time.time() - t0),
                       "mean_prompt_tokens": sum(r["prompt_tokens"] for r in recs) / len(recs)}, open(path, "w"))
        print(f"[{d}] {args.name} top{args.topk} {args.langs}: {len(recs)} queries, failed {failed}, "
              f"wall {time.time() - t0:.0f}s", flush=True)


def stage_score(args):
    for proto in ("pooled", "english"):
        table = {}
        for d in args.datasets:
            base = f"{args.out}/{d}"
            if not os.path.exists(f"{base}/emb_top100.json"):
                continue
            qs, qrels = load_meta(d)
            lang = {str(q["query_id"]): q["language"] for q in qs}
            systems = {"Qwen3-VL-Embedding-2B": json.load(open(f"{base}/emb_top100.json"))}
            for f in sorted(glob.glob(f"{base}/*.json")):
                if not os.path.basename(f).startswith(("emb_", "summary")):
                    systems[os.path.basename(f)[:-5]] = json.load(open(f))["ranking"]
            for name, runs in systems.items():
                qids = [q for q in runs if q in qrels and (proto == "pooled" or lang[q] == "english")]
                if qids:
                    table.setdefault(name, {})[d] = {"ndcg10": sum(ndcg10(runs[q], qrels[q]) for q in qids) / len(qids),
                                                     "n": len(qids)}
        for name, per in table.items():
            per["_mean"] = sum(v["ndcg10"] for v in per.values()) / len(per)
        json.dump(table, open(f"{args.out}/summary_{proto}.json", "w"), indent=1)
        print(f"== {proto}")
        for name, per in table.items():
            print(f"  {name:40s} {100 * per['_mean']:.2f} over {len(per) - 1} datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["emb", "rr", "rerank", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--qwen-repo", help="emb, rr: a checkout of github.com/QwenLM/Qwen3-VL-Embedding")
    ap.add_argument("--model", help="emb, rr: model name or path; rerank --backend chat: the served model name")
    ap.add_argument("--tokenizer", help="rerank --backend chat: tokenizer name or path (default: --model)")
    ap.add_argument("--name", help="rr: short name (2B); rerank: run name used in output files")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--langs", default="all", help="all | english | another language")
    ap.add_argument("--batch", type=int, default=16, help="emb: pages per batch")
    ap.add_argument("--backend", choices=["chat", "jev"], default="chat")
    ap.add_argument("--url", help="rerank: server root (key from $VLLM_API_KEY)")
    ap.add_argument("--readout", choices=["listwise", "pointwise"], default="listwise")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
    ap.add_argument("--limit", type=int, help="first N queries per dataset (a smoke test: no result file)")
    args = ap.parse_args()
    if args.stage in ("emb", "rr") and not (args.qwen_repo and args.model):
        ap.error(f"{args.stage} needs --qwen-repo and --model")
    if args.stage in ("rr", "rerank") and not args.name:
        ap.error(f"{args.stage} needs --name")
    if args.stage == "rerank":
        if not args.url or (args.backend == "chat" and not args.model):
            ap.error("rerank needs --url, and --model with --backend chat")
        if not 2 <= args.topk <= MAX_OPTIONS:
            ap.error(f"rerank: --topk must be between 2 and {MAX_OPTIONS}")
    {"emb": stage_emb, "rr": stage_rr, "rerank": stage_rerank, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
