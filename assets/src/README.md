# Figure sources

`how_it_works.png` and `listwise_prompt.png` are rendered from the `.tex` files of the same name (TikZ, pdfLaTeX with
the `noto` fonts; rendering needs `pip install pymupdf`):

```bash
pdflatex how_it_works.tex
python -c "import fitz; fitz.open('how_it_works.pdf')[0].get_pixmap(dpi=365).save('../how_it_works.png')"
pdflatex listwise_prompt.tex
python -c "import fitz; fitz.open('listwise_prompt.pdf')[0].get_pixmap(dpi=384).save('../listwise_prompt.png')"
```

The colours, the gradient and the mark are in `glancerank-style.tex`.
