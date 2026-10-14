# Limitations

- **One evaluated use.** The technical report measures reranking for multimodal retrieval. `Decider.choice` and
  `Decider.yes_no` run other decisions through the same readout, and we report no numbers for them.
- **At most 16 options.** The labels are A–P, so a choice has 2 to 16 options, and a reranker refines the top 16 of a
  first-stage list in one prompt. The published Qwen3-VL-Reranker numbers rerank the top 100.
- **Option order.** A listwise readout can change its answer when the options are reordered; AnyJev, Runoff and
  others measure this for text options. We present candidates in first-stage order and have not measured the effect
  for image candidates.
- **Model size.** The zero-shot models in the report have 27B parameters; the supervised rerankers have 2B and 8B.
  Smaller VLMs used zero-shot as listwise rerankers have been reported to fall below the first stage.
- **Long prompts.** Sixteen full-resolution document pages are about 29k input tokens, so the server needs a long
  context (`--max-model-len 65536` in our runs) and enough memory for it.
- **Labels must be single tokens.** `ChatBackend` checks that each label is one token in the model's tokenizer and
  raises `LabelTokenError` otherwise.
- **No video.** Video is not supported as a state or an option.
