# Project 1 report

[Read the report](project1-report.pdf) or edit [the LaTeX source](project1-report.tex).
The report includes the six assignment sections, measured result tables, eight
figures, training dynamics, the 5000-level residual ablation, and source references.

From the repository root, compile with Tectonic:

```bash
tectonic report/project1-report.tex --outdir report
```

Alternatively, run `pdflatex project1-report.tex` twice from `report/`.
The source uses standard LaTeX packages and reads figures from
`../outputs/report-figures/`. No training is needed to rebuild the report.

`project1-report-source.zip` contains the LaTeX file and its eight image assets
with portable image paths under `figures/`. Extract the archive before compiling,
or upload it to Overleaf and select `project1-report.tex` as the main document. The older
IEEE template files are retained separately and are not the submitted report.

The tested grading commit and final run receipt are recorded in `SUBMISSION.md`.
Retrospective log intentions remain labeled. The report does not present missing
measurements as estimates or count continued weights as fresh training.
