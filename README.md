# Artefacts — Maintaining a Temporal Knowledge Cube over Streaming Short Text

**Dimension Evolution and Mergeable Semantic Measures**

This repository is the **supplemental material** for the ICDE 2027 research-track
submission. It holds three things:

1. **`supplementary/`** — the compiled supplement (14 pp.), containing the six
   tables and three figures omitted from the twelve-page body for space, plus
   Appendix A (design decisions), Appendix B (reproducibility checklist) and
   Appendix C (the E13 annotation programme in full). Citation numbers in the
   supplement are identical to the main paper.
2. **`experiments/`** — one directory per experiment: the released script, the
   raw machine-readable outputs, and a written report naming the exact
   discrepancy checks that were run.
3. **`data/`** — the data manifest: where each public dataset came from, its
   licence, and the date each query was issued.

Nothing here is new evidence relative to the paper. Every result appears in the
paper in summarised form; the artefacts exist so the numbers can be
**re-derived** rather than taken on trust.

---

## 1. Repository map

```
.
├── README.md                    this file
├── LICENSE                      MIT (code); datasets keep their own licences
├── CITATION.cff                 machine-readable citation
├── supplementary/
│   ├── supplementary.pdf        compiled supplement (14 pp.)
│   ├── supplementary.tex        source of the above
│   ├── appendix_src.tex         Appendix A/B/C source
│   ├── floats/01..09.tex        the nine omitted tables/figures, one file each
│   ├── figures/                 the three figures referenced by the supplement
│   ├── refs_icde.tex            reference list (numbers identical to the paper)
│   └── tools/                   the two build scripts (need the paper source)
├── experiments/                 one directory per experiment
└── data/README.md               data manifest, licences, snapshot dates
```

Experiment directories keep their original names; a gloss follows each one in
the table below. Section numbers refer to the main paper.

| Directory | Paper section | What it holds |
|---|---|---|
| `E1_增量vs全量重建` (incremental vs. full rebuild) | §7.2, E1 | 13-checkpoint equivalence test, cost ratios |
| `E1_维度演化注入` (dimension-evolution injection) | §7.2, E1-EVO | affected-cell share and cost of evolution |
| `E1_滑动窗迟到事件` (sliding windows / late events) | §7.2, E1-SW | window-breach rates under Δw |
| `E2_空间维消歧` (geographic disambiguation) | §7.3, E2 | CED `user.location` → province/city codes |
| `E2_跨期正确性` (as-of correctness across versions) | §7.3, E2 | as-of vs. static ρ, CTAmap change log |
| `E3_源独立性验收` (source-independence acceptance) | §7.4.3, E3 | S1/S2/S3 features, AUC and Cliff's δ |
| `E4_PHEME画像` (PHEME profiling) | §7.4.3 | carrier reconnaissance that fixed the design |
| `E5_P2真实撤回` (real retractions) | §7.4.1, E5 | 393 real retractions, incremental ≡ full |
| `E6_Q5失效正确率` (invalidation correctness) | §7.4.1, E6 | exact vs. approximate lineage, the accuracy trap |
| `E7_端到端串联` (end-to-end pipeline) | §7.5, E7 | the seven independent cross-checks |
| `E8_Wikidata声明级撤回` (Wikidata retraction) | §7.6, E8 | 16 properties, deprecation reasons |
| `E9_修订史裁定时刻` (adjudication time from history) | §7.6, E9 | 62-statement boundary-verified sample |
| `E10_Baseline3_语义消融` (semantic ablation) | §7.7, E10 | numeric cube vs. layered measures |
| `E11_Q3异常预警` (alert criterion) | §7.4.3, E11 | stratified 5-fold + temporal split |
| `E12_D8样本量重定` (sparsity threshold) | §7.8, E12 | derived n_min(ε), six real grids |
| `E13_Q4血缘准确率` (lineage and the semantic layer) | §7.4.2, E13 | protocol, codebook, truth files, scorer |
| `E14_Baseline2_EraRAG` (published RAG baseline) | §7.7, E14 | cost/structure/stability, patch file |
| `E15_话题侧演化真值与缝处置` (topic-side ground truth) | §7.9 / §8 | the negative result on topic granularity |
| `E16_P1物理成本` (physical cost of P1) | §7.2, E16 | wall-clock, affine memory, repeatability |
| `决策点3_度量表示` (measure representation decision) | §5 | the layered-representation argument |
| `可合并度量代数` (mergeable measure algebra) | §5 | T1–T6 statements and proofs |

Each directory follows the same shape: a `*_报告.md` (report) stating what was
measured and which checks had to pass, one or more `*.py` scripts, and the raw
outputs (`*.json` / `*.csv`) they produced. **The report is the entry point.**

---

## 2. Requirements

* **Python 3.9+; no GPU is required for any experiment.** The whole paper was
  produced on a CPU-only machine, and the published baseline (E14) was
  deliberately run in the same configuration.
* Dependencies are the standard library plus `requests` / `pandas` /
  `scikit-learn` per script; each script states its own imports at the top.
* Network access is needed only for the Wikidata experiments (E8/E9), which
  query the public SPARQL endpoint. Everything else runs on the downloaded
  snapshots listed in `data/README.md`.
* An LLM API key is needed only for E14, and the experiment is designed so that
  **no absolute token or wall-clock figure is claimed**: every cost is a ratio
  to a full rebuild measured on the same host in the same run.

---

## 3. Reproducing an experiment

1. Open the experiment's report (`*_报告.md`) — it states the claim, the
   procedure, and the exact checks that must pass.
2. Run the script(s) in that directory; each writes its raw output next to
   itself.
3. Compare against the committed raw outputs.

The checks named in the reports are the **acceptance gate, not a formality**.
E1's thirteen checkpoints, E7's seven cross-checks, E10's cell-for-cell
zero-difference requirement and E9's boundary verification (the predecessor
revision must not already be deprecated) are all machine-checked; any of them
failing means the reproduction has not succeeded.

---

## 4. The published baseline (E14, EraRAG) and its patch

E14 exercises the public `EraRAG-Official` checkout on its own incremental path.
**The released code contained 13 defects that had to be repaired locally before
it would run to completion.** The list, the reasoning for each, and a `git diff`
patch are released as `E14_repo_compat.patch` and `E14_repo_compat.md` inside
that directory. The upstream checkout itself is **not** redistributed here.

Two consequences, stated in the paper and repeated because they bound what the
result means:

* The defects are defects of *this implementation*. The result reads as "this
  incremental path is not a safe one", not as a proof that hierarchical-summary
  maintenance is unimplementable; the general claim is algebraic (T6, P4).
* The stability result is re-tested in a **process-per-batch** mode in which each
  batch reloads the persisted tree before refining, so a crash cannot be
  attributed to in-memory state.

---

## 5. What is deliberately *not* in this repository

* **Raw third-party data.** PHEME, CED, CTAmap, Wikidata, Weibo21 and WeiboHAN
  are not redistributed; `data/README.md` gives the canonical source, the
  licence, and the checksum of the snapshot actually used.
* **The upstream EraRAG checkout** (see §4) and any downloaded model weights.
* **Local caches** (`*.pkl`, Hugging Face cache) and **runtime configuration
  directories containing credentials**. Scripts regenerate these on first run;
  a stub config template is committed in place of the live one.

---

## 6. Licence and citation

Code and text in this repository are released under the **MIT licence**
(`LICENSE`). The datasets keep their own licences, listed in `data/README.md`;
nothing here re-distributes them.

If these artefacts help your work, please cite the paper — see `CITATION.cff`.
