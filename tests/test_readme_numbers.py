"""The README's result tables are transcribed from bench/results/*.json; this keeps them in step.

A cell shows the JSON mean to two decimals once the run is complete (every task or dataset), and "—" before that.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
MMEB = json.loads((ROOT / "bench/results/mmeb_v2.json").read_text(encoding="utf-8"))["systems"]
VIDORE = json.loads((ROOT / "bench/results/vidore_v3.json").read_text(encoding="utf-8"))["systems"]
LATENCY = json.loads((ROOT / "bench/results/latency.json").read_text(encoding="utf-8"))["results"]


def table(header_start):
    """Rows (lists of cells) of the README table whose header row starts with `header_start`."""
    lines = README.splitlines()
    start = next(i for i, x in enumerate(lines) if x.startswith(header_start))
    rows = []
    for x in lines[start + 2:]:
        if not x.startswith("|"):
            break
        rows.append([re.sub(r"&nbsp;|<sup>.*?</sup>", "", c).strip() for c in x.strip().strip("|").split("|")])
    assert rows, header_start
    return rows


def expected(entry):
    return "%.2f" % entry["mean"] if entry and entry["complete"] else "—"


def test_main_table():
    rows = table("| reranker | trained for reranking |")
    assert [r[0] for r in rows] == ["Qwen3-VL-Embedding-2B", "Qwen3-VL-Reranker-2B", "Qwen3-VL-Reranker-8B",
                                    "VLM-as-Jev, Qwen3.8-27B", "VLM-as-Jev, JEV-27B-VL"]
    for name, _, image, visdoc, vidore in rows:
        assert image == expected(MMEB.get(name, {}).get("image")), (name, "image")
        assert visdoc == expected(MMEB.get(name, {}).get("visdoc")), (name, "visdoc")
        assert vidore == expected(VIDORE.get(name, {}).get("pooled")), (name, "vidore pooled")


def test_vidore_english_table():
    for name, cell in table("| reranker | ViDoRe V3 English |"):
        assert cell == expected(VIDORE.get(name, {}).get("english")), name


def test_readout_table():
    first_stage = expected(MMEB["Qwen3-VL-Embedding-2B"]["image"])
    assert f"the first stage scores {first_stage})" in README
    for backbone, listwise, pointwise in table("| backbone | listwise, 1 prompt |"):
        assert listwise == expected(MMEB.get(f"VLM-as-Jev, {backbone}", {}).get("image")), backbone
        assert pointwise == expected(MMEB.get(f"{backbone}, pointwise", {}).get("image")), backbone


def test_latency_table():
    rows = table("| reranker | model calls |")
    assert len(rows) == len(LATENCY)
    for name, calls, seconds in rows:
        assert seconds == "%.2f" % LATENCY[name]["mean_s"], name
        assert int(calls) == round(LATENCY[name]["calls_per_query"]), name
