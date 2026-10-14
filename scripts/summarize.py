"""Collect per-task results into the JSON files under bench/results/, which the README tables are read from.

    python scripts/summarize.py --out bench/results \\
        --mmeb "Qwen3-VL-Embedding-2B=runs/mmeb/Qwen3-VL-Embedding-2B:first_stage" \\
        --mmeb "VLM-as-Jev, Qwen3.8-27B=runs/mmeb/base_listwise" \\
        --vidore-dir runs/vidore3 --vidore "VLM-as-Jev, Qwen3.8-27B=base_listwise_top16_all" \\
        --latency "VLM-as-Jev, Qwen3.8-27B=runs/latency/latency_base_listwise.json:base_listwise"

--mmeb NAME=DIR[:first_stage]  DIR/<modality>/<task>_rerank_score.json (or <task>_score.json for the first stage)
--vidore NAME=RUN              RUN as named in DIR/summary_{pooled,english}.json (written by eval_vidore3.py score)
--latency NAME=FILE:KEY        one row of a latency.py output
Values are in percent (MMEB, ViDoRe) or seconds (latency); every task found is kept, with its count.
"""
import argparse
import glob
import json
import os

METRIC = {"image": "hit@1", "visdoc": "ndcg_linear@5"}
N_TASKS = {"image": 12, "visdoc": 24}
N_DATASETS = 8


def spec(s):
    name, _, rest = s.partition("=")
    if not name or not rest:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {s!r}")
    return name, rest


def mmeb_tasks(path, modality):
    first_stage = path.endswith(":first_stage")
    d = path[: -len(":first_stage")] if first_stage else path
    suffix = "_score.json" if first_stage else "_rerank_score.json"
    out = {}
    for f in sorted(glob.glob(f"{d}/{modality}/*{suffix}")):
        task = os.path.basename(f)[: -len(suffix)]
        if first_stage and task.endswith("_rerank"):
            continue
        out[task] = round(100 * float(json.load(open(f))[METRIC[modality]]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmeb", type=spec, action="append", default=[])
    ap.add_argument("--vidore-dir")
    ap.add_argument("--vidore", type=spec, action="append", default=[])
    ap.add_argument("--latency", type=spec, action="append", default=[])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mmeb:
        systems = {}
        for name, path in args.mmeb:
            systems[name] = {}
            for modality in METRIC:
                tasks = mmeb_tasks(path, modality)
                if tasks:
                    systems[name][modality] = {"metric": METRIC[modality], "n_tasks": len(tasks),
                                               "complete": len(tasks) == N_TASKS[modality],
                                               "mean": round(sum(tasks.values()) / len(tasks), 4), "tasks": tasks}
        json.dump({"benchmark": "MMEB-V2 retrieval, top-16 of Qwen3-VL-Embedding-2B", "unit": "percent",
                   "systems": systems}, open(f"{args.out}/mmeb_v2.json", "w"), indent=1)

    if args.vidore:
        summaries = {p: json.load(open(f"{args.vidore_dir}/summary_{p}.json")) for p in ("pooled", "english")}
        systems = {}
        for name, run in args.vidore:
            systems[name] = {}
            for proto, table in summaries.items():
                per = {d: v for d, v in table.get(run, {}).items() if d != "_mean"}
                if per:
                    ds = {d: round(100 * v["ndcg10"], 4) for d, v in per.items()}
                    systems[name][proto] = {"n_datasets": len(ds), "complete": len(ds) == N_DATASETS,
                                            "queries": sum(v["n"] for v in per.values()),
                                            "mean": round(sum(ds.values()) / len(ds), 4), "datasets": ds}
        json.dump({"benchmark": "ViDoRe V3, nDCG@10, top-16 of Qwen3-VL-Embedding-2B",
                   "protocols": {"pooled": "all six languages pooled per dataset, mean over datasets",
                                 "english": "English queries only, mean over datasets"},
                   "unit": "percent", "systems": systems}, open(f"{args.out}/vidore_v3.json", "w"), indent=1)

    if args.latency:
        rows, meta = {}, None
        for name, ref in args.latency:
            path, _, key = ref.rpartition(":")
            d = json.load(open(path))
            meta = meta or d["meta"]
            rows[name] = d["results"][key]
        json.dump({"meta": meta, "unit": "seconds per query", "results": rows}, open(f"{args.out}/latency.json", "w"),
                  indent=1)


if __name__ == "__main__":
    main()
