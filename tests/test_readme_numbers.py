"""The README's result tables are transcribed from bench/results/*.json; this keeps them in step.

A cell shows the JSON mean to two decimals once the run is complete (every task or dataset), and "—" before that.
"""
import json
import re
from decimal import ROUND_HALF_EVEN, Decimal
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
        cells = [re.sub(r"&nbsp;|<sup>.*?</sup>|\*\*", "", c).strip() for c in x.strip().strip("|").split("|")]
        if any(cells[1:]):  # group rows such as "*Supervised rerankers*" have no values
            rows.append(cells)
    assert rows, header_start
    return rows


def show(v):
    """Two decimals, round half to even (as the report's tables): 64.675 -> 64.68, 71.425 -> 71.42."""
    return str(Decimal(f"{v:.6f}").quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def expected(entry):
    return show(entry["mean"]) if entry and entry["complete"] else "—"


MAIN_ROWS = {"Qwen3-VL-Embedding-2B (first stage)": "Qwen3-VL-Embedding-2B",
             "Qwen3-VL-Reranker-2B": "Qwen3-VL-Reranker-2B", "Qwen3-VL-Reranker-8B": "Qwen3-VL-Reranker-8B",
             "JEV-27B-VL, listwise (text decision tuning)": "JEV-27B-VL, listwise",
             "Qwen3.8-27B, listwise (no decision tuning)": "Qwen3.8-27B, listwise",
             **{f"Gemma-4-{g}, listwise": f"Gemma-4-{g}, listwise" for g in ("E2B", "E4B", "12B", "26B-A4B", "31B")}}
LISTWISE = {"Qwen3.8-27B": "Qwen3.8-27B, listwise", "JEV-27B-VL": "JEV-27B-VL, listwise"}


def test_main_table():
    rows = table("| | MMEB-V2 image")
    assert [r[0] for r in rows] == list(MAIN_ROWS)
    for row, image, visdoc, vidore in rows:
        name = MAIN_ROWS[row]
        assert image == expected(MMEB.get(name, {}).get("image")), (name, "image")
        assert visdoc == expected(MMEB.get(name, {}).get("visdoc")), (name, "visdoc")
        assert vidore == expected(VIDORE.get(name, {}).get("pooled")), (name, "vidore pooled")


def test_readout_table():
    first_stage = expected(MMEB["Qwen3-VL-Embedding-2B"]["image"])
    assert f"the first stage scores {first_stage}" in README
    for row, listwise, pointwise in table("| backbone | listwise · 1 prompt |"):
        backbone = row.split(" (")[0]  # "Qwen3.8-27B (no decision tuning)" -> "Qwen3.8-27B"
        assert listwise == expected(MMEB.get(LISTWISE[backbone], {}).get("image")), backbone
        assert pointwise == expected(MMEB.get(f"{backbone}, pointwise", {}).get("image")), backbone


def test_latency_table():
    rows = table("| reranker | model calls |")
    assert len(rows) == len(LATENCY)
    for name, calls, tokens, seconds in rows:
        assert seconds == "%.2f" % LATENCY[name]["mean_s"], name
        assert tokens == "%.1fk" % (LATENCY[name]["mean_prefill_tokens"] / 1000), name
        assert int(calls) == round(LATENCY[name]["calls_per_query"]), name
