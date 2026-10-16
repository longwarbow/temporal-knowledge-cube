# Supplementary material and artefacts

**Maintaining a Temporal Knowledge Cube over Streaming Short Text:
Dimension Evolution and Mergeable Semantic Measures**

This repository is the supplemental material for the ICDE 2027 submission.
It contains three things:

1. **`supplementary.pdf`** — the six tables and three figures that were omitted
   from the twelve-page body for space (their numbers are summarised inline in
   the main paper), plus Appendix A (design decisions), Appendix B
   (reproducibility checklist) and Appendix C (the E13 annotation programme in
   full). Citation numbers in `supplementary.pdf` are identical to the main
   paper.
2. **The experiment artefacts** — for every experiment the released script, the
   raw machine-readable outputs, and a written report stating the exact
   discrepancy checks that were run.
3. **The data manifest** — where each public dataset came from, its licence,
   and the date every query was issued.

Nothing in this repository is new evidence relative to the paper: every result
appears in the paper in summarised form, and the artefacts exist so that the
numbers can be re-derived rather than taken on trust.

---

## 1. Repository map

```
supplement/
├── README.md                  this file
├── supplementary.pdf          compiled supplement (14 pp.)
├── supplementary.tex          source of the above
├── appendix_src.tex           Appendix A/B/C, generated from the paper source
├── floats/01..09.tex          the nine omitted tables/figures, one file each
├── figures/                   the three figures referenced by the supplement
├── refs_icde.tex              reference list (numbers identical to the paper)
└── experiments/               copied from 实验/, one directory per experiment
```

Experiment directories keep their original (Chinese) names; a gloss follows
each one below. Section numbers refer to the main paper.

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

Each directory follows the same shape: a `*_报告.md` (report) that states what
was measured and which checks had to pass, one or more `*.py` scripts, and the
raw outputs (`*.json` / `*.csv`) they produced. The report is the entry point;
it names the exact invariants (13 checkpoints in E1, 7 cross-checks in E7,
cell-for-cell 0-difference in E10, boundary verification in E9) whose failure
would have stopped the experiment.

---

## 2. Requirements

* Python 3.9+; no GPU is required for any experiment. The whole paper was
  produced on a CPU-only machine, and the published baseline (E14) was
  deliberately run in the same configuration.
* Dependencies are standard-library-plus-`requests`/`pandas`/`scikit-learn`
  per script; each script states its own imports at the top.
* Network access is needed only for the Wikidata experiments (E8/E9), which
  query the public SPARQL endpoint. Everything else runs on the downloaded
  snapshots listed in §3.
* An LLM API key is needed only for E14 (the published baseline), and the
  experiment is designed so that **no absolute token or wall-clock figure is
  claimed** — every cost is reported as a ratio to a full rebuild measured on
  the same host in the same run.

---

## 3. Data sources and licences

| Dataset | Role | Source | Licence / access |
|---|---|---|---|
| CED | repost/anomaly stream (1,278,493 events) | public GitHub archive (codeload) | research use |
| CTAmap | versioned geographic dimension, 450 real division changes (2000–2025) | Zenodo, DOI `10.5281/zenodo.22179207` | CC-BY-4.0 |
| PHEME | claim-state ground truth (6,425 threads, 105,747 tweets) | Hugging Face mirror + jsDelivr for the annotation files | released for research |
| Wikidata | declaration-level retraction truth | public SPARQL endpoint (WDQS) | CC0 |
| Weibo21 / WeiboHAN | domain gold standards and cross-checks | public releases | research use |

All retrieval is scripted and channel-pinned. Where a platform's TLS stack was
intercepted on our hosts, the scripts shell out to `curl` with an explicit
`--noproxy '*'`, retry with a documented policy, and send a User-Agent carrying
a contact address. No proprietary API is a dependency of any reported number.

**Snapshot caveat (important).** Wikidata is a live graph. The whole-graph
count of `rank=deprecated` statements moved by four statements within twenty
minutes during profiling. Every Wikidata number in the paper must be read
together with its query date, which is stamped in the artefacts; the counts
reported in §7.6 describe the graph as of 2026-10-11.

---

## 4. Reproducing an experiment

1. Open the experiment's report (`*_报告.md`) — it states the claim, the
   procedure and the exact checks that must pass.
2. Run the script(s) in that directory; each writes its raw output next to
   itself.
3. Compare against the committed raw outputs. The checks named in the report
   are the acceptance gate, not a formality: E1's thirteen checkpoints, E7's
   seven cross-checks, E10's cell-for-cell zero-difference requirement and E9's
   boundary verification (the predecessor revision must not already be
   deprecated) are all machine-checked, and any of them failing means the
   reproduction has not succeeded.

---

## 5. The published baseline (E14, EraRAG) and its patch

E14 exercises the public `EraRAG-Official` checkout on its own incremental
path. **The released code contained 13 defects that had to be repaired locally
before it would run to completion**; the list, the reasoning for each, and a
`git diff` patch are released as `E14_repo_compat.patch` and
`E14_repo_compat.md` in that directory.

Two consequences are stated explicitly in the paper and are repeated here
because they bound what the result means:

* The defects are defects of *this implementation*. The result reads as "this
  incremental path is not a safe one", not as a proof that hierarchical-summary
  maintenance is unimplementable; the general claim is algebraic (T6, P4).
* The stability result is re-tested in a **process-per-batch** mode in which
  each batch reloads the persisted tree before refining, so a crash cannot be
  attributed to in-memory state.

---

## 6. Licences and citation

Code in this repository is released under the MIT licence. The datasets keep
their own licences, listed in §3; nothing here re-distributes them, the
manifest points to the canonical source.

If this repository helps your work, please cite the paper.

---

## 7. Before you publish this repository

- [ ] Copy the 21 experiment directories from `实验/` into `experiments/`.
- [ ] Confirm no directory contains an API key, a token, or a private path
      (E14 used a free-tier key; check `E14_环境与key验证_*.md` in particular).
- [ ] Put the compiled `supplementary.pdf` at the repository root, and give the
      repository a stable tag (`v1.0-submission`) so the paper's URL resolves
      to the exact state that was reviewed.
- [ ] Add the repository URL to the submission form's supplemental-material
      field — ICDE 2027 reviews its availability as part of the evaluation.
