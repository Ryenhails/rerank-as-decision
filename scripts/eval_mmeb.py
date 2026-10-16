"""Rerank the MMEB-V2 retrieval tasks (image, visdoc) with the listwise readout, under the protocol of
Qwen3-VL-Reranker's own evaluation (github.com/QwenLM/Qwen3-VL-Embedding at 393e297):

- candidates: the first --topk ids of Qwen3-VL-Embedding-2B's ``<task>_pred.jsonl`` (the file their reranker reads);
- queries, candidates and per-query instructions: the repo's dataset loaders;
- metrics: the repo's ``RankingMetrics``; output ``<task>_rerank_score.json`` / ``_rerank_pred.jsonl`` in the same
  format, so the repo's ``gather_results.py`` averages them.

Only the model call differs: one listwise read over all K <= 16 candidates (or K pointwise reads), no decoding.
Every call is logged to ``<out>/<task>_calls.jsonl``; a rerun resumes from it.

    python scripts/eval_mmeb.py --qwen-repo Qwen3-VL-Embedding --data-basedir mmeb_v2 \\
        --config Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/image_retrieval.yaml \\
        --encode-output-path outputs/Qwen3-VL-Embedding-2B/image --out results/base_listwise/image \\
        --backend chat --url http://127.0.0.1:8000 --model Qwen/Qwen3.8-27B --readout listwise
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

from listwise_rerank import MAX_CANDIDATES, ChatBackend, JevDecideBackend, Ranker
from listwise_rerank.ranker import DEFAULT_INSTRUCTION


def make_ranker(args):
    if args.backend == "jev":
        backend = JevDecideBackend(args.url)
    else:
        backend = ChatBackend(args.url, model=args.model, tokenizer=args.tokenizer)
    return Ranker(backend, readout=args.readout, max_pixels=args.max_pixels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-repo", required=True, help="a checkout of github.com/QwenLM/Qwen3-VL-Embedding")
    ap.add_argument("--data-basedir", required=True, help="the MMEB-V2 media directory the repo's configs point into")
    ap.add_argument("--config", required=True, help="one of the repo's *_retrieval.yaml")
    ap.add_argument("--tasks", nargs="*", help="a subset of the config's tasks (default: all)")
    ap.add_argument("--encode-output-path", required=True, help="Qwen3-VL-Embedding-2B outputs for this modality")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["chat", "jev"], default="chat")
    ap.add_argument("--url", required=True, help="server root, e.g. http://127.0.0.1:8000 (key from $VLLM_API_KEY)")
    ap.add_argument("--model", help="chat backend: the served model name")
    ap.add_argument("--tokenizer", help="chat backend: tokenizer name or path (default: --model)")
    ap.add_argument("--readout", choices=["listwise", "pointwise"], default="listwise")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
    ap.add_argument("--workers", type=int, default=8, help="queries in flight")
    ap.add_argument("--limit", type=int, help="first N queries per task (a smoke test: nothing is written but the log)")
    args = ap.parse_args()
    if not 2 <= args.topk <= MAX_CANDIDATES:
        ap.error(f"--topk must be between 2 and {MAX_CANDIDATES}")
    if args.backend == "chat" and not args.model:
        ap.error("--backend chat needs --model")
    sys.path.insert(0, args.qwen_repo)
    import yaml
    from src.evaluation.mmeb_v2.arguments import DataArguments, RerankArguments
    from src.evaluation.mmeb_v2.data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
    from src.evaluation.mmeb_v2.eval_reranker import build_corpus_lookup, load_topk_from_pred
    from src.evaluation.mmeb_v2.utils.eval_utils.metrics import RankingMetrics

    ranker = make_ranker(args)
    os.makedirs(args.out, exist_ok=True)
    configs = yaml.safe_load(open(args.config))
    for task, cfg in configs.items():
        if args.tasks and task not in args.tasks:
            continue
        score_path = f"{args.out}/{task}_rerank_score.json"
        if os.path.exists(score_path) and not args.limit:
            print(f"[{task}] done earlier, skipping", flush=True)
            continue
        t_task = time.time()
        topk_ids = load_topk_from_pred(os.path.join(args.encode_output_path, f"{task}_pred.jsonl"), args.topk)
        for k in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
            if cfg.get(k):
                cfg[k] = os.path.join(args.data_basedir, cfg[k])
        qry, corpus = AutoEvalPairDataset.instantiate(model_args=RerankArguments(model_name_or_path="listwise_rerank"),
                                                      data_args=DataArguments(dataset_config=args.config), **cfg)
        cands = generate_cand_dataset(qry, corpus)
        qry = qry.add_column("topk_ids", topk_ids)
        lookup = build_corpus_lookup(cands)
        n = len(qry) if not args.limit else min(args.limit, len(qry))
        calls_path = f"{args.out}/{task}_calls.jsonl"
        done = {}
        if os.path.exists(calls_path):
            for line in open(calls_path):
                r = json.loads(line)
                done[r["i"]] = r
        lock = threading.Lock()
        fh = open(calls_path, "a")

        def one(i):
            if i in done:
                return done[i]
            item = qry[i]
            q = dict(item["query_input"])
            instruction = q.pop("instruction", None) or DEFAULT_INSTRUCTION
            ids = [d for d in item["topk_ids"] if d in lookup][: args.topk]
            rec = {"i": i, "ids": ids, "probs": None, "prompt_tokens": 0, "calls": 0, "seconds": 0.0, "error": None}
            if len(ids) >= 2:
                t = time.time()
                for _ in range(3):
                    try:
                        r = ranker.rank(q, [lookup[d] for d in ids], instruction=instruction)
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
            recs = list(ex.map(one, range(n)))
        fh.close()
        preds = []
        for i, rec in enumerate(recs):
            info = qry[i]["dataset_infos"]
            gt = info["label_name"] if isinstance(info["label_name"], list) else [info["label_name"]]
            ids = rec["ids"]
            if rec["probs"]:
                p = dict(zip(ids, rec["probs"]))
                order = sorted(ids, key=lambda d: (-p[d], ids.index(d)))
                scores = [p[d] for d in order]
            else:  # failed or fewer than 2 candidates: keep the first-stage order
                order, scores = ids, []
            preds.append({"label": gt, "rel_scores": info.get("rel_scores"), "retrieved": ids, "prediction": order,
                          "rerank_scores": scores})
        metrics = RankingMetrics(cfg.get("metrics", ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]))
        sd = metrics.evaluate(preds)
        sd.update(num_pred=len(preds), num_data=len(preds), topk=args.topk,
                  reranker_model=f"listwise_rerank backend={ranker.backend.name} readout={args.readout}",
                  failed=sum(1 for r in recs if not r["probs"]),
                  mean_prompt_tokens=sum(r["prompt_tokens"] for r in recs) / max(1, len(recs)),
                  wall_seconds=round(time.time() - t_task, 1))
        if not args.limit:
            json.dump(sd, open(score_path, "w"), indent=4)
            with open(f"{args.out}/{task}_rerank_pred.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
        main_keys = [k for k in sd if k in ("hit@1", "ndcg_linear@5")]
        print(f"[{task}] n={len(preds)} " + " ".join(f"{k}={sd[k]:.4f}" for k in main_keys)
              + f" failed={sd['failed']} tok/q={sd['mean_prompt_tokens']:.0f} wall={sd['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
