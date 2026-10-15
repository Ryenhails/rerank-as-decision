# Changelog

## 0.1.0 (unreleased)

First public version, released with the technical report *VLMs as Zero-Shot System One Rerankers for Multimodal
Retrieval*.

- `Ranker.rank(query, candidates)`: up to 16 candidates (images, document pages, text or a mix) in one prompt under the
  labels A–P, ranked by the next-token probabilities of the labels (listwise); or one yes/no question per candidate
  (pointwise). Its prompts and requests are byte-identical to the ones behind the report's numbers.
- The project was called VLM-as-Jev in its first drafts; it is now GlanceRank (package `glancerank`).
- Backends: `ChatBackend` (any OpenAI-compatible chat endpoint with log-probabilities, tested with vLLM 0.30.0),
  `JevDecideBackend` (JEV-27B-VL's `/v1/decide`), `FakeBackend` (tests).
- `ChatBackend` raises `LabelTokenError` when the server returns no log-probability for a label. The report's runs
  read a missing label as probability 0; no label was missing in any of them.
- Scripts: `eval_mmeb.py`, `eval_vidore3.py`, `latency.py`, `summarize.py`; results in `bench/results/`.
- `patches/qwen3vl_reranker_tf5_compat.patch`: runs Qwen3-VL-Reranker's released code under transformers 5.
- README with the cover, a figure of the readout (`assets/how_it_works.png`, TikZ source in `assets/src/`) and the
  results tables.
