# Limitations

- **One evaluated setting.** The technical report measures reranking for multimodal retrieval: a first-stage
  retriever followed by the choice readout on its top 16.
- **At most 16 candidates.** The labels are A–P, so one prompt ranks 2 to 16 candidates, and the readout refines the top
  16 of a first-stage list. The published Qwen3-VL-Reranker numbers rerank the top 100.
- **Option order.** A listwise readout can change its answer when the options are reordered, and it can prefer some
  labels whatever the input; AnyJev and Runoff measure this for text options. We present candidates in first-stage order and have not measured the effect
  for image candidates.
- **Model size and generation.** The main zero-shot models in the report have 27B parameters and a newer backbone
  (Qwen3.8); the supervised rerankers have 2B and 8B and are built on Qwen3-VL. On the Gemma 4 family the two smallest models (E2B, E4B) fall below the first stage on all three benchmarks.
- **Image budget across families.** The Gemma 4 runs keep the processor's default of 280 visual tokens per image (the
  model supports up to 1120), against up to 1800 for the Qwen models, so comparisons between the two families also change the image resolution.
- **Long prompts.** Sixteen full-resolution document pages are about 29k input tokens, so the server needs a long
  context (`--max-model-len 65536` in our runs) and enough memory for it.
- **Labels must be single tokens.** `ChatBackend` checks that each label is one token in the model's tokenizer and
  raises `LabelTokenError` otherwise.
- **No video.** Video is not supported as a state or an option.
