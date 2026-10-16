# Third-party models, data and code

Nothing below is redistributed in this repository; the scripts download or read it at run time.

| What | Where | License | Used for |
|---|---|---|---|
| Qwen3.8-27B | https://huggingface.co/Qwen/Qwen3.8-27B | Apache-2.0 | listwise readout (no tuning) |
| JEV-27B-VL | https://huggingface.co/autotrust/JEV-27B-VL | Apache-2.0 (model card) | comparison: decision-tuned Jev-style model, `/v1/decide` |
| Qwen3-VL-Embedding-2B | https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B | Apache-2.0 | first-stage retriever |
| Qwen3-VL-Reranker-2B, -8B | https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B, https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B | Apache-2.0 | supervised baselines |
| Qwen3-VL-Embedding code | https://github.com/QwenLM/Qwen3-VL-Embedding (commit `393e297`) | Apache-2.0 | MMEB-V2 loaders and metrics, the models' released inference code; `patches/` modifies it |
| MMEB-V2 | https://huggingface.co/datasets/TIGER-Lab/MMEB-V2, https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct | not stated on the dataset cards | image and visual-document retrieval; evaluation only |
| ViDoRe V3 | https://huggingface.co/datasets/vidore (`vidore_v3_<dataset>`, 8 public datasets) | CC-BY-4.0 | document page retrieval |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 | serving |
