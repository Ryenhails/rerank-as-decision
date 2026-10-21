# rerank-as-decision

<p>
  <a href="https://arxiv.org/abs/2610.28435"><img alt="arXiv 2610.28435" src="https://img.shields.io/badge/arXiv-2610.28435-D96570"></a>
  <a href="https://ryenhails.github.io/rerank-as-decision/"><img alt="Project page" src="https://img.shields.io/badge/project-page-4285F4"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-34A853"></a>
  <a href="https://github.com/Ryenhails/rerank-as-decision/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Ryenhails/rerank-as-decision/actions/workflows/ci.yml/badge.svg"></a>
</p>

Code of the paper [*Multimodal Reranking Is a System One Decision*](https://arxiv.org/abs/2610.28435)
([project page](https://ryenhails.github.io/rerank-as-decision/)). The paper takes the position that a
general vision-language model (VLM) can serve as a multimodal reranker through System One decisions, and that decision
tuning can improve its reranking. Reranking is one such decision: the state holds the query and its candidates, the
question asks which candidate best matches the query, and the options are the candidates.

This package asks that question of an off-the-shelf VLM. The query and up to 16 candidates go into one prompt under the
labels **A–P**, and the next-token probabilities of the labels are the ranking: one prefill, no decoding, no training
for reranking. The readout is that of setwise prompting (Zhuang et al., 2024) and FIRST (Reddy et al., 2024), applied
to images and pages. The report uses it to rerank the top 16 of a first-stage retriever.

## How it works

<p align="center"><img src="assets/how_it_works.png" width="100%" alt="One prompt with the query and the labelled candidates; one forward pass of the VLM; the next-token probabilities of the labels A to P, renormalized, give the new order"></p>

1. **One prompt.** The query and every candidate go into one prompt, each candidate under a label from A to P.
2. **One forward pass.** The model reads the prompt once and generates nothing.
3. **Label probabilities.** The next-token distribution, restricted to the label tokens and renormalized, gives one
   probability per candidate. The candidates are sorted by it.

Because all candidates share one prompt, the model compares them with each other: this is the *choice* readout. The
*yes/no* readout asks one question per candidate instead, the form of supervised pointwise rerankers such as
Qwen3-VL-Reranker; asked zero-shot, it falls below the first stage in the report.

<p align="center"><img src="assets/listwise_prompt.png" width="100%" alt="The choice prompt: a user turn with the instruction, the query and sixteen page images labelled A to P, then the question; an assistant turn with thinking off, where the next-token logits of A to P are read and renormalized into one probability per candidate"></p>

The prompt, simplified: one user turn holds the instruction, the query and the candidates, each after its label, and
the question. The assistant turn opens with thinking off, and the next-token logits at that position are read for the
labels A–P. Chat-template tokens are omitted; `rerank_as_decision/ranker.py` builds the exact prompt.

## Quickstart

```bash
pip install -e ".[client]"
vllm serve Qwen/Qwen3.8-27B --logprobs-mode processed_logprobs \
    --max-model-len 65536 --limit-mm-per-prompt '{"image": 20}'
```

```python
from PIL import Image
from rerank_as_decision import ChatBackend, Ranker

ranker = Ranker(ChatBackend("http://127.0.0.1:8000", model="Qwen/Qwen3.8-27B"))

pages = [Image.open(f"page_{i}.png") for i in range(16)]          # a first-stage top-16, best first
r = ranker.rank("What was the operating margin in 2023?", pages,
                instruction="Find a document image that matches the given query.")
r.order      # candidate indices, best first
r.scores     # one probability per candidate, in input order; they sum to 1
```

A query or a candidate is a string, an image, a dict with `"image"` and `"text"`, or a list of these. Images are
downscaled to 1800 × 32 × 32 pixels. `Ranker(..., readout="pointwise")` uses the yes/no readout, one question per
candidate. JEV-27B-VL, a Jev-style decision model on the same backbone, can be queried through its own decision head
with `JevDecideBackend("http://127.0.0.1:8000")`.

## Results

From the paper, in brief:

- **Question type.** Asked one choice over all candidates, a general VLM improves the first stage; asked a yes/no
  question per candidate, it falls below it (MMEB-V2 image retrieval).
- **Zero-shot against dedicated rerankers.** Through one zero-shot decision, Qwen3.8-27B beats Qwen3-VL-Reranker-2B on
  all three benchmarks, comes within 1.6 points of Reranker-8B on each, and matches it on MMEB-V2 visual documents.
- **Decision tuning.** JEV-27B-VL, the same backbone tuned on text decisions only, reranks pages slightly better on
  most tasks and natural images no better.
- **Scale.** Among Gemma 4 models, those of 12B parameters and more improve page retrieval zero-shot; the two smaller
  ones lower it.

The zero-shot model is larger than the supervised rerankers and of a newer generation, so these comparisons are
confounded; the report discusses this and the experiments that would test the position.

Every reranker refines the same top-16 of Qwen3-VL-Embedding-2B and receives the same images; Gemma 4 keeps its
processor's default of 280 visual tokens per image (the others use up to 1800).

| | MMEB-V2 image<br><sub>hit@1 · 12 tasks</sub> | MMEB-V2 visdoc<br><sub>nDCG@5 · 24 tasks</sub> | ViDoRe V3<br><sub>nDCG@10 · 8 datasets</sub> |
|:--|:--:|:--:|:--:|
| Qwen3-VL-Embedding-2B (first stage) | 74.85 | 79.20 | 53.01 |
| *Supervised rerankers* | | | |
| Qwen3-VL-Reranker-2B | 74.49 | 82.72 | 59.18 |
| Qwen3-VL-Reranker-8B | 78.36 | 84.78 | 61.98 |
| *Choice readout, one prefill* | | | |
| **Qwen3.8-27B, choice** (no decision tuning) | 76.76 | 84.79 | 60.94 |
| JEV-27B-VL, choice (text decision tuning) | 76.80 | 85.42 | 62.03 |
| *Gemma 4, choice readout (no decision tuning)* | | | |
| Gemma-4-E2B, choice | 64.68 | 68.87 | 42.21 |
| Gemma-4-E4B, choice | 62.03 | 78.60 | 51.89 |
| Gemma-4-12B, choice | 73.53 | 81.81 | 55.62 |
| Gemma-4-26B-A4B, choice | 73.11 | 82.14 | 56.40 |
| Gemma-4-31B, choice | 77.41 | 80.85 | 55.42 |

<sub>ViDoRe V3 pools its six languages within each dataset.</sub>

**Choice or yes/no, with or without decision tuning** (MMEB-V2 image, hit@1; the first stage scores 74.85)

| backbone | choice · 1 prompt | yes/no · 16 prompts |
|:--|:--:|:--:|
| Qwen3.8-27B (no decision tuning) | 76.76 | 69.50 |
| JEV-27B-VL (text decision tuning) | 76.80 | 71.42 |

<details>
<summary><b>Latency</b> on one H200</summary>

<br>

Wall time per query from request to scores, preprocessing included, at concurrency 1: the same top-16 ViDoRe V3 pages
for 100 English queries. Input tokens are summed over a query's model calls. Qwen3-VL-Reranker runs with its released
code, which scores the 16 pairs one after another; the 27B and Gemma 4 models run on vLLM 0.30.0. Scoring the 16 pairs
as one padded batch, a variant measured for the report, brings Reranker-2B and -8B down to 1.68 and 2.61 s, so one
choice pass of Qwen3.8-27B takes 1.1 times as long as the released Reranker-8B and 1.5 times as long as a batched one.

| reranker | model calls | input tokens / query | seconds / query |
|:--|:--:|:--:|:--:|
| Qwen3-VL-Reranker-2B | 16 | 29.6k | 2.75 |
| Qwen3-VL-Reranker-8B | 16 | 29.6k | 3.67 |
| Qwen3.8-27B, choice | 1 | 28.8k | 3.97 |
| JEV-27B-VL, choice | 1 | 28.9k | 4.48 |
| Qwen3.8-27B, yes/no | 16 | 29.8k | 3.10 |
| JEV-27B-VL, yes/no | 16 | 29.9k | 3.57 |
| Gemma-4-E2B, choice | 1 | 4.4k | 0.47 |
| Gemma-4-E4B, choice | 1 | 4.4k | 0.49 |
| Gemma-4-12B, choice | 1 | 4.4k | 0.55 |
| Gemma-4-26B-A4B, choice | 1 | 4.4k | 0.78 |
| Gemma-4-31B, choice | 1 | 4.4k | 1.27 |

</details>

Every number in the tables above is read from [`bench/results/`](bench/results), which also holds the per-task
scores, and `tests/test_readme_numbers.py` checks the tables against it. Commands, versions and run times:
[docs/reproduce.md](docs/reproduce.md). Known limits: [docs/limitations.md](docs/limitations.md).

## Contributing

Issues and pull requests are welcome. A backend is one file with two methods; see [CONTRIBUTING.md](CONTRIBUTING.md)
and [ROADMAP.md](ROADMAP.md).

## Citation

```bibtex
@misc{liu2026multimodalrerankingdecision,
  title         = {Multimodal Reranking Is a System One Decision},
  author        = {Zhuchenyang Liu and Yao Zhang and Yu Xiao},
  year          = {2026},
  eprint        = {2610.28435},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {https://arxiv.org/abs/2610.28435}
}
```

## License

Apache-2.0. Models and datasets keep their own licenses ([THIRD_PARTY.md](THIRD_PARTY.md)).

<sub>An independent research project, not affiliated with TypeSafe AI (Jev) or autotrust (JEV-27B-VL).</sub>
