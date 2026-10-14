# Roadmap

Done:

- [x] `choice` and `yes_no` decisions over images, pages and text, read in one forward pass
- [x] Reranking with the readout; results on MMEB-V2 image and visual-document retrieval and on ViDoRe V3
- [x] vLLM chat backend and JEV-27B-VL `/v1/decide` backend

Next, open to contributions:

- [ ] **help wanted** In-process `transformers` backend (no server), with a parity check against `ChatBackend`
- [ ] **help wanted** SGLang backend
- [ ] **help wanted** Order robustness: average over cyclic shifts of the options, and measure how often the answer
      changes when the options are reordered (for text options, AnyJev reports this for LLMs)
- [ ] More than 16 options: a tournament or sliding window over groups of 16
- [ ] `score` decisions (ordered levels) next to `choice` and `yes_no`
- [ ] Video as a state or an option
- [ ] Results beyond reranking: visual question routing, image classification, document triage
