# Reproducing the technical report

Every number in the report and the README comes from the steps below. Each run used one NVIDIA H200.

## Versions

| component | version |
|---|---|
| Qwen3-VL-Embedding code | github.com/QwenLM/Qwen3-VL-Embedding at `393e297` |
| Qwen3-VL-Embedding-2B | Hugging Face revision `9f2f7e7` |
| Qwen3-VL-Reranker-2B / -8B | revisions `4bd860a` / `b212dc8` |
| JEV-27B-VL (bundles the unchanged Qwen3.8-27B weights) | revision `fee8c1d` |
| Qwen side | torch 2.8, transformers 5.8, flash-attn 2.7.3, datasets 4.8 |
| 27B models | vLLM 0.30.0 |

## 1. First stage: Qwen3-VL-Embedding-2B

**MMEB-V2.** In the Qwen repository, with the MMEB-V2 media under `$MMEB`:

```bash
for M in image visdoc; do
  torchrun --nproc_per_node=1 -m src.evaluation.mmeb_v2.eval_embedding --normalize true \
      --per_device_eval_batch_size 16 --model_name_or_path Qwen/Qwen3-VL-Embedding-2B \
      --dataset_config scripts/evaluation/mmeb_v2/${M}_retrieval.yaml \
      --encode_output_path outputs/Qwen3-VL-Embedding-2B/$M/ --data_basedir $MMEB
done
```

This writes `<task>_pred.jsonl` (the full ranking) and `<task>_score.json` per task. Our means: image 74.85, visdoc
79.20 (published: 74.8 and 79.2).

**ViDoRe V3.**

```bash
python scripts/eval_vidore3.py emb --out runs/vidore3 --qwen-repo Qwen3-VL-Embedding \
    --model Qwen/Qwen3-VL-Embedding-2B
```

Pooling all six languages per dataset and averaging over the eight datasets gives 53.0 (published: 52.9).

## 2. Supervised rerankers: Qwen3-VL-Reranker-2B / -8B at top-16

Under transformers 5, first apply the patch to the Qwen repository (without it the released reranker fails in
`get_rope_index`; under transformers 4.57 the patch changes nothing):

```bash
cd Qwen3-VL-Embedding && git apply ../GlanceRank/patches/qwen3vl_reranker_tf5_compat.patch
```

**MMEB-V2**, in the Qwen repository:

```bash
for R in 2B 8B; do for M in image visdoc; do
  torchrun --nproc_per_node=1 -m src.evaluation.mmeb_v2.eval_reranker --per_device_eval_batch_size 16 \
      --model_name_or_path Qwen/Qwen3-VL-Reranker-$R \
      --dataset_config scripts/evaluation/mmeb_v2/${M}_retrieval.yaml \
      --encode_output_path outputs/Qwen3-VL-Embedding-2B/$M/ \
      --rerank_output_path outputs/Qwen3-VL-Reranker-${R}_top16/$M/ --data_basedir $MMEB --topk 16
done; done
```

**ViDoRe V3:**

```bash
for R in 2B 8B; do
  python scripts/eval_vidore3.py rr --out runs/vidore3 --qwen-repo Qwen3-VL-Embedding \
      --model Qwen/Qwen3-VL-Reranker-$R --name $R --topk 16 --langs all
done
```

On one H200 this took 2.0 s per query for the 2B model and 2.9 s for the 8B model (14,514 queries).

## 3. GlanceRank and JEV-27B-VL

**Server.** Our runs served JEV-27B-VL with the server in its repository (`serve_decide.py`, vLLM 0.30.0). The same
server answers chat requests for the model name `autotrust/JEV-27B-VL` with the unchanged Qwen3.8-27B weights, and
`/v1/decide` with the decision head. The flags:

```bash
export VLLM_API_KEY=<a random token>
python JEV-27B-VL/serve_decide.py --model JEV-27B-VL --served-model-name autotrust/JEV-27B-VL \
    --enable-lora --max-lora-rank 32 --lora-modules jev-decision=JEV-27B-VL/adapter_vllm \
    --logprobs-mode processed_logprobs --max-model-len 65536 --enable-prefix-caching \
    --mamba-cache-mode align --max-num-seqs 8 --trust-request-chat-template \
    --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 80}' \
    --host 127.0.0.1 --port 8000
```

We used `{"image": 80}` for MMEB-V2 and `{"image": 20}` for ViDoRe V3. A stock `vllm serve Qwen/Qwen3.8-27B --logprobs-mode processed_logprobs` exposes the same chat interface; we
have not compared its numbers with these.

**The four configurations** (backend × readout):

| report | `--backend` | `--model` | `--readout` |
|---|---|---|---|
| GlanceRank (Qwen3.8-27B) | `chat` | `autotrust/JEV-27B-VL` | `listwise` |
| JEV-27B-VL, decision head | `jev` | | `listwise` |
| Qwen3.8-27B, pointwise | `chat` | `autotrust/JEV-27B-VL` | `pointwise` |
| JEV-27B-VL, pointwise | `jev` | | `pointwise` |

With `--backend chat`, the label token ids come from the tokenizer of `--model` (here the Qwen3.8 tokenizer in the
JEV-27B-VL repository); `--tokenizer` points to a local copy instead.

**MMEB-V2:**

```bash
for M in image visdoc; do
  python scripts/eval_mmeb.py --qwen-repo Qwen3-VL-Embedding --data-basedir $MMEB \
      --config Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/${M}_retrieval.yaml \
      --encode-output-path Qwen3-VL-Embedding/outputs/Qwen3-VL-Embedding-2B/$M \
      --out runs/mmeb/jev_listwise/$M --backend jev --url http://127.0.0.1:8000 --readout listwise
done
```

**ViDoRe V3:**

```bash
python scripts/eval_vidore3.py rerank --out runs/vidore3 --backend jev --url http://127.0.0.1:8000 \
    --readout listwise --name jev_listwise --topk 16 --langs all
python scripts/eval_vidore3.py score --out runs/vidore3
```

Both scripts send 8 queries at a time (`--workers 8`) and log every call, so an interrupted run resumes where it
stopped. On one H200, the listwise readout took about 1.3 hours for the 12,000 MMEB-V2 image queries.

## 4. Latency

```bash
python scripts/latency.py qwen --vidore-out runs/vidore3 --qwen-repo Qwen3-VL-Embedding \
    --models 2B=Qwen/Qwen3-VL-Reranker-2B 8B=Qwen/Qwen3-VL-Reranker-8B --out runs/latency
python scripts/latency.py vlm --vidore-out runs/vidore3 --backend jev --url http://127.0.0.1:8000 \
    --readout listwise --name jev_listwise --out runs/latency
```

The server for the latency runs used `--mm-processor-cache-gb 0`, so that a later configuration does not reuse the
image preprocessing of an earlier one. 55 English queries from each of two datasets (computer_science, hr); the first
10 are dropped as warm-up, which leaves 100.

## 5. Tables

`scripts/summarize.py` collects the runs into `bench/results/*.json` (its docstring shows the arguments), and the
README tables are read from those files.
