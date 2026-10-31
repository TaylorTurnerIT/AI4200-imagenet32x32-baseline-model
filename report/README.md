# Project 1 report

[Read the report](project1-report.pdf) or edit [the LaTeX source](project1-report.tex).
The report uses the IEEE Transactions journal layout through the standard
`IEEEtran` class. It retains the six assignment sections, measured result tables,
eight figures, training dynamics, the 5000-level residual ablation, and references.

From the repository root, compile with Tectonic:

```bash
tectonic report/project1-report.tex --outdir report
```

Alternatively, run `pdflatex project1-report.tex` twice from `report/`.
The source requires the standard `IEEEtran` class and reads figures from
`../outputs/report-figures/`. No training is needed to rebuild the report.

`project1-report-source.zip` contains the LaTeX file and its eight image assets
with portable image paths under `figures/`. Extract the archive before compiling,
or upload it to Overleaf and select `project1-report.tex` as the main document. The older
IEEE TAI template files are retained separately; this report uses `IEEEtran`.

The tested grading commit and final run receipt are recorded in `SUBMISSION.md`.
Retrospective log intentions remain labeled. The report does not present missing
measurements as estimates or count continued weights as fresh training.

The author revisions were committed before formatting. `verification.json` records
the exact content comparison: all prose, six tables, eight images and captions,
and the title and author text were preserved. Only layout commands changed.
