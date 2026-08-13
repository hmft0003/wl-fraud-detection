# 1-WL Structural Signatures vs. Graph Convolutional Networks for Fraud Detection

MSc Data Science project (COMP5200M), School of Computing, University of Leeds.
Supervisor: Dr Sebastian Ordyniak.

## Research question

Does 1-Weisfeiler-Leman (1-WL) neighbourhood structure add predictive signal
beyond node-level transaction features for illicit-transaction detection, and at
what computational cost, using a Graph Convolutional Network (GCN) as a learned
baseline?

The Elliptic Bitcoin dataset is the object of study. Cora serves as a positive
control: a citation network on which graph signal is known to be strong, used to
establish that the pipeline extracts structural signal when structural signal is
present.

## Summary of findings

**On predictive performance, a well-diagnosed negative result.** The relaxed 1-WL
histogram improves on the local transaction features by +0.0059 AUPRC
(95% CI [+0.0035, +0.0081]) — statistically significant, but small. Structural
controls confirm the gain is genuinely attributable to the graph rather than to
the extra feature columns: removing every edge returns the score to baseline, and
randomising the edge endpoints pushes it below baseline. The signature recovers
roughly half of what the dataset authors' hand-crafted neighbourhood aggregates
provide, does not match them, and adds nothing once they are present. The
discrete formulation over-discriminates and falls below baseline entirely.

**On computational cost, a positive result.** The 1-WL pipeline is
non-parametric and trains roughly six times faster than the GCN, which carries
10,754 trained parameters. Where a graph method is used on this dataset at all,
the cheap signature is preferable to the expensive network.

**Why the graph contributes so little.** Not a lack of homophily — adjusted for
class prior, Elliptic (0.737) and Cora (0.768) are comparable. The distinguishing
factors are sparsity (79% of Elliptic nodes have at most two neighbours) and
feature strength (the local features already reach 0.7816 AUPRC unaided). 72 of
the 165 supplied features are themselves one-hop neighbour aggregates, so the
baseline already encodes much of what 1-WL computes. The identical pipeline
recovers a large gain on Cora, so the small result is a property of the data
rather than of the method.

### Headline numbers

| Condition | Elliptic (AUPRC) | Cora (accuracy) |
|---|---|---|
| Features only (baseline) | 0.7816 | 0.5643 |
| + 1-WL discrete | 0.7807 | — |
| + 1-WL histogram | 0.7858 | 0.7067 |
| + 1-WL histogram, best K | 0.7875 | 0.7182 |
| + authors' aggregated features | 0.7934 | — |
| GCN | 0.6299 | 0.8090 |

### Reproduction of published benchmarks

| Metric | This work | Published | Source |
|---|---|---|---|
| GCN illicit F1 (Elliptic) | 0.6208 | 0.628 | Weber et al. (2019) |
| Random Forest illicit F1 | 0.8124 | 0.796 | Weber et al. (2019) |
| GCN accuracy (Cora) | 0.8090 | ≈0.81 | Kipf & Welling (2017) |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── wl_vs_gcn_fraud_detection.ipynb   # all experiments
│   └── figures/                          # generated on run
└── data/                                 # NOT tracked — see below
```

## Data

Neither dataset is included in this repository: the Elliptic files are large and
its licence restricts redistribution.

**Elliptic Bitcoin dataset** — download from
[Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place
these three files in `data/` at the repository root:

```
data/elliptic_txs_features.csv
data/elliptic_txs_classes.csv
data/elliptic_txs_edgelist.csv
```

203,769 transactions over 49 time steps; 234,355 edges; 165 features
(93 local + 72 aggregated). 4,545 illicit, 42,019 licit, 157,205 unlabelled.

**Cora** downloads automatically on first run via PyTorch Geometric, into
`notebooks/data/Planetoid/`. Both `data/` directories are excluded by
`.gitignore`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 or later. PyTorch Geometric may need to be installed against your
specific PyTorch and CUDA build — see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if `pip install` alone fails.

## Reproducing the results

The notebook reads Elliptic via the relative path `../data`, so **run it from
the `notebooks/` directory**:

```bash
cd notebooks
jupyter lab wl_vs_gcn_fraud_detection.ipynb
```

Then run all cells top to bottom. A fixed seed (42) is set for NumPy and PyTorch,
and every model is given an explicit `random_state`, so all reported metrics are
reproducible.

Expect on the order of an hour on a laptop CPU. The two quantisation sweeps and
the GCN's epoch-selection pass dominate the runtime.

**Timings are machine-dependent.** The costs reported in the computational-cost
section were measured on an idle CPU; the ratio between the GCN and the
signature pipeline is the stable quantity, not the absolute seconds.

## Verification

The notebook is self-checking rather than self-reporting. Assertions run at the
point each graph is loaded and halt execution rather than let a bad state
propagate:

- Node, edge, and feature counts are checked against the published figures.
- Train, validation, and test masks are asserted disjoint, with cardinalities
  matching the temporal split (29,894 / 16,670 for Elliptic; 140 / 500 / 1,000
  for Cora).
- The quantiser is asserted to receive local features only, never labels.
- The colour-pruning frequency counts are asserted to be drawn from the training
  period alone.

Three further checks are reported in the notebook output:

- **Active-column diagnostic.** With the true edges present all 320 histogram
  columns are active; with the edges removed only the 64 initial-colour columns
  survive and all 186 neighbour-mean columns become identically zero. This
  converts the edgeless control from an assertion into a mechanical
  demonstration.
- **Structural controls.** Edgeless (0.7811) and randomised-edge (0.7801)
  variants against the full graph (0.7858) and baseline (0.7816).
- **Two-stage GCN evaluation.** The epoch count is selected on a validation
  window drawn from inside the training period (t = 30–34), after which the
  network is retrained from scratch on the full training period. The test set is
  never used for model selection.

## Method notes

- **Temporal split.** Training on time steps 1–34 and testing on 35–49, following
  Weber et al. (2019), as a deployed detector would. No random shuffling.
- **Metric.** AUPRC is the primary metric, suited to the 9.8% positive class;
  illicit-class F1 is reported alongside for comparability with published work.
- **Significance.** Differences are assessed by a paired bootstrap over 1,000
  resamples of the test nodes, reporting a 95% confidence interval on the paired
  difference in AUPRC.
- **Transductive fitting.** Colour vocabularies are fitted over all nodes, with
  no labels, matching the setting in which the GCN operates. A deployed system
  could not cluster on future transactions, so the reported figures are an
  optimistic estimate in that specific sense.

## Known limitations

- The Random Forest and the GCN differ in both classifier and representation, so
  only the within-classifier comparisons isolate the contribution of the graph.
- The test window contains the dark-market shutdown at time step 43, a concept
  drift that Weber et al. report degrades every method and which is not modelled
  here.
- The negative result is established on a single transaction graph; its
  generality to denser or actor-level graphs is open.

## References

- [Weber et al. (2019)](https://arxiv.org/abs/1908.02591). *Anti-Money Laundering
  in Bitcoin: Experimenting with Graph Convolutional Networks for Financial
  Forensics.* KDD Workshop on Anomaly Detection in Finance.
- [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907). *Semi-Supervised
  Classification with Graph Convolutional Networks.* ICLR.
- [Xu et al. (2019)](https://arxiv.org/abs/1810.00826). *How Powerful are Graph
  Neural Networks?* ICLR.
- [Corso et al. (2020)](https://arxiv.org/abs/2004.05718). *Principal
  Neighbourhood Aggregation for Graph Nets.* NeurIPS.
