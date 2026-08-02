# Dashboards (optional)

Self-contained interactive HTML visualizations, one per experiment. They read
no external data (the numbers are inlined), so each opens directly in a browser.

Included here:

- `level3_dashboard.html` — E7 sequential knowledge editing (the flagship result).

The remaining per-experiment dashboards from the study
(`encoding_dashboard.html`, `sparse_dashboard.html`,
`generalization_dashboard.html`, `attention_dashboard.html`,
`capacity_dashboard.html`, `level0_dashboard.html`, `level2_dashboard.html`,
and the overview `dashboard.html`) are optional presentation artifacts; drop
them into this folder if you want the full set in the release. They are not
required to reproduce any result — every figure in the paper is regenerated
from `results/` by `paper/figures/make_figures.py`.
