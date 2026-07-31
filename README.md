# 1-WL Structural Signatures vs. Graph Convolutional Networks for Fraud Detection

MSc project (COMP5200M), University of Leeds.

## Research question

Does 1-Weisfeiler-Leman (1-WL) neighbourhood structure add predictive signal
beyond node-level transaction features for fraud detection, and how does it
compare with a Graph Convolutional Network (GCN) baseline?

## Summary of findings

1-WL recovers roughly half of the available neighbourhood signal on both
datasets tested. On the Elliptic Bitcoin dataset that signal is small the
graph is sparse (mean degree 2.3) and the node features are already highly
informative so no neighbourhood aggregation gains much. On Cora, used as a
positive control where graph signal is known to be strong, the same pipeline
recovers a large share of the gap to the GCN, confirming the method is sound.
The result on Elliptic is therefore a property of the data, not of the method.

The GCN reproduces the illicit F1 of 0.628 reported by Weber et al. (2019); the
Random Forest and the Cora GCN likewise reproduce published figures.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── wl_vs_gcn_fraud_detection.ipynb   # main experiments
│   └── figures
└── data/                                 # NOT tracked — see below
```

## Data

The datasets are not included in this repository (large, and the Elliptic
licence restricts redistribution).

**Elliptic Bitcoin dataset** - download from Kaggle [Elliptic Data set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place these three files in `data/`:

```
data/elliptic_txs_features.csv
data/elliptic_txs_classes.csv
data/elliptic_txs_edgelist.csv
```

**Cora** downloads automatically on first run via PyTorch Geometric.

## Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/wl_vs_gcn_fraud_detection.ipynb
```

Run the notebook top to bottom. A fixed random seed (42) makes every result
reproducible.

## References

- Weber et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with
  Graph Convolutional Networks for Financial Forensics.* KDD Workshop on
  Anomaly Detection in Finance.
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph
  Convolutional Networks.* ICLR.
- Xu et al. (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Corso et al. (2020). *Principal Neighbourhood Aggregation for Graph Nets.*
  NeurIPS.
