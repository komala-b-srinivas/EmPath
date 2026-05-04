# EmPath LaTeX Report — Compilation Guide

## Files

| File | Purpose |
|------|---------|
| `empath_report.tex` | Main document (~12,000 words, APA 7th edition) |
| `references.bib` | BibTeX database (40+ references) |
| `README_latex.md` | This file |

---

## Compile on Overleaf (Recommended — Zero Setup)

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project → Upload Project**
2. Zip both files together: `empath_report.tex` + `references.bib`
3. Upload the zip
4. In Overleaf settings (top-left menu): set **Compiler = pdfLaTeX**, **Bibliography = Biber**
5. Click **Recompile**

Overleaf handles all package installation automatically.

---

## Compile Locally

### Prerequisites

Install a full TeX distribution:
- **macOS**: [MacTeX](https://www.tug.org/mactex/) — `brew install --cask mactex`
- **Linux**: `sudo apt install texlive-full`
- **Windows**: [MiKTeX](https://miktex.org/)

Verify required packages are installed:
```
apa7, biblatex, biber, tikz, pgfplots, listings, booktabs, csquotes, babel
```
All are included in `texlive-full` / MacTeX.

### Compile Commands (run in order)

```bash
cd /Users/komalabelursrinivas/Desktop/Capstone/EmPath_v2

pdflatex empath_report.tex   # Pass 1: build aux files
biber empath_report           # Process bibliography
pdflatex empath_report.tex   # Pass 2: resolve citations
pdflatex empath_report.tex   # Pass 3: finalize cross-refs
```

Output: `empath_report.pdf`

### One-liner

```bash
pdflatex empath_report.tex && biber empath_report && pdflatex empath_report.tex && pdflatex empath_report.tex
```

---

## Package Summary

| Package | Purpose |
|---------|---------|
| `apa7` (stu option) | APA 7th edition student paper formatting |
| `biblatex` + `biber` | Bibliography with APA citation style |
| `tikz` + `pgfplots` | System diagrams and data visualizations |
| `listings` | Python code snippets with syntax highlighting |
| `booktabs` | Publication-quality tables |
| `csquotes` + `babel` | Required by biblatex-apa |

---

## Document Structure

| Section | Word Count (approx.) |
|---------|----------------------|
| Abstract | 200 |
| Introduction | 1,400 |
| Related Work | 1,800 |
| Method | 2,800 |
| Results | 1,900 |
| Discussion | 2,200 |
| Limitations & Future Directions | 1,400 |
| Conclusion | 500 |
| **Total** | **~12,200** |

---

## Figures (TikZ — no external images needed)

| Figure | Type | Content |
|--------|------|---------|
| 1 | TikZ flowchart | End-to-end data pipeline |
| 2 | TikZ block diagram | Stacked fusion architecture |
| 3 | TikZ grid | LOSO cross-validation protocol |
| 4 | pgfplots matrix | Confusion matrix |
| 5 | pgfplots xbar | Biosignal SHAP importance |
| 6 | pgfplots xbar | Landmark SHAP importance |
| 7 | pgfplots ybar interval | Per-subject accuracy distribution |

---

## Tables

| Table | Content |
|-------|---------|
| 1 | Dataset composition after reactivity filtering |
| 2 | Complete 35-feature biosignal set |
| 3 | Primary performance metrics vs. baselines |
| 4 | Full 26-variant ablation study |
| 5 | Top-10 SHAP values per modality |
| 6 | Per-subject accuracy tiers |
| 7 | Hyperparameter justification (in Method section) |

---

## Code Snippets

| Snippet | Content |
|---------|---------|
| 1 | GSR slope + Shannon entropy extraction |
| 2 | Person-specific z-score normalization |
| 3 | LOSO evaluation loop |
| 4 | SHAP TreeExplainer computation |

---

## Customization

- **Author / affiliation / date**: Edit `\authorsnames`, `\authorsaffiliations`, `\duedate` in the preamble
- **Add figures from Results/**: Replace TikZ figures with `\includegraphics` if you prefer PNG exports — e.g.:
  ```latex
  \includegraphics[width=\linewidth]{Results/error_analysis_v2/shap_biosignal_bar.png}
  ```
- **Word count**: Run `texcount empath_report.tex` after compilation

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `! LaTeX Error: File 'apa7.sty' not found` | Run `tlmgr install apa7` or use Overleaf |
| `biber: command not found` | Install via `tlmgr install biber` |
| `Package biblatex Warning: Please (re)run Biber` | Run `biber empath_report` then recompile |
| Figures show `?` instead of numbers | Run pdflatex a third time |
| `pgfplots` color issues | Add `\pgfplotsset{compat=1.18}` (already present) |
