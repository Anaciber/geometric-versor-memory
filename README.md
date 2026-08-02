# Geometric Versor Memory and Geometric Attention

**A Clifford-Algebra Substrate for One-Shot Binding, Editing, and Order Encoding**

Ignacio María Ozcáriz Arraiza · ORCID [0000-0002-9511-016X](https://orcid.org/0000-0002-9511-016X) · anaciber@gmail.com

> Reproducible code, data logs, and figures for the paper. The paper studies an
> associative-memory and attention model whose state is a multivector in a
> Clifford geometric algebra and whose operations are geometric products by
> \*versors\*: binding is one geometric product, recall is multiplication by the
> versor inverse, and a stored fact is deleted exactly with no retraining. All
> experiments run on small algebras Cl(n,0), n ≤ 8, with synthetic data.

📄 **Paper:** [`paper/paper.pdf`](paper/paper.pdf)

\---

## What is here

Every number in the paper comes from a script in `src/` and is logged as JSON in
`results/`; the three figures are regenerated from those logs by
`paper/figures/make\_figures.py`. Nothing is hand-entered.

```
geometric-versor-memory/
├── README.md
├── LICENSE                      # MIT (code)
├── CITATION.cff                 # citation metadata (software + paper)
├── requirements.txt             # numpy, scipy, matplotlib, jax
├── .gitignore
├── src/                         # all Python (flat: imports are unchanged)
│   ├── cln.py                   # Cl(n,0) geometric-algebra engine  (core)
│   ├── vam.py                   # versor associative memory          (core)
│   ├── encoding.py              # E1 — single-field capacity \& encoding
│   ├── sparse.py                # E2 — sparse addressable memory
│   ├── generalize.py            # E3 — noise, analogy, interpolation
│   ├── attention.py             # E4 — geometric attention
│   ├── level0.py                # E0 — classical baselines (Hebbian/Hopfield/MLP)
│   ├── level2.py                # E6 — versor position binding in transformers
│   └── level3.py                # E7 — sequential knowledge editing (flagship)
├── results/                     # JSON logs behind every reported number
│   ├── encoding\_results.json  capacity\_curves.json
│   ├── sparse\_results.json
│   ├── generalization\_results.json
│   ├── attention\_results.json
│   ├── level0\_results.json    level0\_noise.json
│   ├── level2\_results.json
│   └── level3\_results.json    level3\_load2.json  level3\_stress.json
├── paper/
│   ├── paper.tex   paper.pdf
│   └── figures/  fig\_editing.pdf  fig\_sparse.pdf  fig\_capacity.pdf  make\_figures.py
└── dashboards/                  # optional interactive HTML dashboards
```

> \*\*Note.\*\* Copy your working `.py` files into `src/` and your `\*\_results.json`
> into `results/` (keep the filenames above so the mapping below holds). The
> Quantum-Market network simulator and the topological (L1) notes are separate
> projects and are intentionally not included here.

## Experiment → script → result map

|Exp.|What it shows|Script|Log|
|-|-|-|-|
|E0|Classical baselines (Hebbian, modern Hopfield, MLP)|`src/level0.py`|`level0\_results.json`, `level0\_noise.json`|
|E1|Single-field capacity; role of the code family|`src/encoding.py`|`encoding\_results.json`, `capacity\_curves.json`|
|E2|Sparse addressable memory → linear scaling|`src/sparse.py`|`sparse\_results.json`|
|E3|Generalization: noise, analogy, interpolation|`src/generalize.py`|`generalization\_results.json`|
|E4|Geometric attention (interpolation, analogy, order)|`src/attention.py`|`attention\_results.json`|
|E6|Versor position binding vs RoPE (trained transformer)|`src/level2.py`|`level2\_results.json`|
|E7|Sequential knowledge editing vs ROME (flagship)|`src/level3.py`|`level3\_results.json`, `level3\_load2.json`, `level3\_stress.json`|

## Install

```bash
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10. The memory experiments (E0–E4) need only `numpy`; the
trained-transformer experiments (E6–E7) additionally use `jax`.

## Reproduce

Run any experiment from inside `src/` (scripts write their JSON to the working
directory):

```bash
cd src
python encoding.py     # E1
python sparse.py       # E2
python generalize.py   # E3
python attention.py    # E4
python level0.py       # E0
python level2.py       # E6   (jax)
python level3.py       # E7   (jax)
```

Regenerate the three paper figures from the committed logs:

```bash
python paper/figures/make\_figures.py     # writes fig\_editing/sparse/capacity.pdf
```

## Build the paper

```bash
cd paper
pdflatex paper.tex \&\& pdflatex paper.tex   # two passes for references
```

The bibliography is embedded (`thebibliography`), so no BibTeX run is needed.

## Citation

If you use this code or the results, please cite the paper (see also
`CITATION.cff`):

```bibtex
@article{ozcariz2026versor,
  title   = {Geometric Versor Memory and Geometric Attention: A Clifford-Algebra
             Substrate for One-Shot Binding, Editing, and Order Encoding},
  author  = {Ozc\\'ariz Arraiza, Ignacio Mar\\'ia},
  doi     = {10.5281/zenodo.21752907},
  url     = {https://doi.org/10.5281/zenodo.21752907},
  year    = {2026}
}
```

## License

Code is released under the MIT License (`LICENSE`). The paper text and figures
are © the author; on arXiv they are distributed under the license selected at
submission (CC BY 4.0 recommended).

