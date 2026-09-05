# NAAN — Network Anomaly Assessment for Nodes

**Razorpay AI Buildathon 2026 — AI Risk Manager track**
Srikumaran S

NAAN detects recipient accounts showing mule-account-like incoming-transaction
patterns in digital payments. It combines unsupervised clustering to surface
anomalies with a dual-signal LLM agent that turns them into auditable,
human-reviewable risk flags — it never approves, blocks, or moves money.

## Problem

RBI's April 2026 discussion paper reports NCRP fraud cases grew from 2.6 lakh
(2021) to 28 lakh (2025). The paper distinguishes account-takeover fraud
(largely mitigated) from **Authorised Push-Payment (APP) fraud**, and
references RBI's own **Mulehunter.AI** initiative for detecting mule accounts
used to launder fraudulent funds — a framing this project mirrors.

## Architecture

```
Raw PaySim transactions
        │
        ▼
Feature engineering (per nameDest / recipient)
  avg_amount, std_amount, max_avg_ratio,
  txn_count, txn_per_active_day, dispersion_index
        │
        ├──────────────┬───────────────────────┐
        ▼              ▼                       ▼
   DBSCAN            XGBoost              XGBoost
 (unsupervised    (account-level,     (transaction-level,
  anomaly signal)  negative result)    ROC-AUC 0.939)
        │              │                       │
        └──────┬───────┘                       │
               ▼                                │
     Dual-signal LLM agent (Groq, gpt-oss-120b) │
     — flags disagreement between signals ──────┘
               │
               ▼
   Structured risk assessment → human reviewer
   (risk_level, explanation, recommended_action, confidence)
```

## Key results

| Stage | Result |
|---|---|
| DBSCAN (eps=0.30) | Best silhouette (0.475); noise cluster shows 2.06% fraud vs 0.82% baseline (~2.5x enrichment) |
| Account-level XGBoost | ROC-AUC 0.556 — negative result, reported honestly |
| Transaction-level XGBoost | ROC-AUC 0.939, recall 0.83 @ default threshold |
| Dual-signal agent | Surfaces disagreement between DBSCAN and classifier instead of averaging/hiding it |

## Repo structure

```
NAAN/
├── README.md
├── DS-Phase.ipynb          # cleaning, features, EDA, PCA, DBSCAN/K-Means/Agglomerative
├── XGBoost.ipynb           # account-level + transaction-level classifiers
└── LLM_Connect.ipynb       # single-signal → dual-signal Groq agent
```

## Setup

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn xgboost joblib kagglehub groq
```

Notebooks were developed in **Google Colab with a T4 GPU** using cuDF/cuML
(RAPIDS) for feature engineering and clustering, with a validated CPU/pandas
fallback path. cuML/cuDF are not preinstalled on a fresh Colab runtime and
must be reinstalled per session (see DS-Phase.ipynb, cell 2).

Dataset: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle,
`ealaxi/paysim1`), loaded via `kagglehub.dataset_download("ealaxi/paysim1")`.

Trained artifacts (`fitted_scaler.pkl`, `fitted_pca.pkl`, `core_points.npy`,
`fraud_classifier_xgb.pkl`, `txn_level_classifier_xgb.pkl`) are expected under
a `PROJECT_DIR` (default: `/content/drive/MyDrive/fraud_detection_project/`)
so scoring new accounts doesn't require retraining — see LLM_Connect.ipynb for
the plug-and-play scoring flow.

`GROQ_API_KEY` must be set as an environment variable to run the agent
notebook.

## Design principle

The agent layer never approves, blocks, or moves money. It only produces a
structured risk assessment for a human reviewer, creating an auditable trail
— NAAN is a triage/priority signal, not a stand-alone automated
decision-maker.

## Pitch video

[link here]
