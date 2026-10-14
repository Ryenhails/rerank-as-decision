# Figure sources

`how_it_works.png` is rendered from `how_it_works.tex` (TikZ, pdfLaTeX with the `noto` fonts; rendering needs `pip install pymupdf`):

```bash
pdflatex how_it_works.tex
python -c "import fitz; fitz.open('how_it_works.pdf')[0].get_pixmap(dpi=365).save('../how_it_works.png')"
```

The colours, the gradient and the mark are in `vlmjev-style.tex`.
