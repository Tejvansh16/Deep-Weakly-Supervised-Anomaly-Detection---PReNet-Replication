# Deep Weakly-Supervised Anomaly Detection — PReNet Replication

Replication study of **PReNet** ([Pang, Shen, Jin & van den Hengel, KDD '23](https://dl.acm.org/doi/10.1145/3580305.3599295)), which reformulates anomaly detection as a pairwise relation prediction problem rather than fitting individual anomaly examples directly.

Individual project for CIS-5700 (Machine Learning), Winter 2026 — Tejvansh Singh Randhawa.

## What this repo contains

- `PReNet_Replication.ipynb` — end-to-end notebook: environment setup, compatibility patches, main experiment, ablation, and training-curve visualization
- `ablation.py` — standalone script for the no-pairwise-supervision baseline
- `training_curves.py` — standalone script for the training loss plot
- `report/` — full project report and presentation slides

## Summary

The official [PReNet codebase](https://github.com/mala-lab/PReNet) targets TensorFlow 1.14 / Python 3.6. This project:

1. **Ported the codebase to TensorFlow 2.19 / Python 3.12**, patching three breaking API changes:
   - Model checkpoint format (`.h5` → `.weights.h5`)
   - `model.fit_generator()` → `model.fit()` with a `tf.data.Dataset` wrapper
   - `keras.backend` loss ops → `tf.reduce_mean` / `tf.abs`
2. **Replicated PReNet on the Annthyroid dataset** (7,200 instances, 21 features, 7.4% anomaly rate) across 5 seeds, using K=30 known outliers (half of the paper's K=60).
3. **Ran an ablation** — the same encoder trained with plain binary cross-entropy instead of pairwise relation supervision — to isolate what the pairwise formulation actually contributes.

## Results

| Method | AUC-ROC | AUC-PR |
|---|---|---|
| Paper PReNet (K=60, 10 runs) | 0.781 | 0.298 |
| **Our PReNet (K=30, 5 runs)** | **0.8012 ± 0.0052** | **0.2918 ± 0.0060** |
| Ablation — no pairwise supervision (K=30, 5 runs) | 0.5102 ± 0.0335 | 0.0935 ± 0.0047 |
| iForest, unsupervised (from paper) | 0.679 | 0.144 |

- The replication **matched (and slightly exceeded)** the paper's reported AUC-ROC using **half the labeled anomalies**, consistent with PReNet's sample-efficiency claims.
- The ablation collapses to near-random performance (AUC-ROC ≈ 0.51), confirming that **pairwise relation learning — not just the encoder architecture — is the key driver** of PReNet's performance.

![Training loss curve](report/training_curve.png)

## Setup / how to run

```bash
git clone https://github.com/mala-lab/PReNet.git
pip install -r requirements.txt
```

Open `PReNet_Replication.ipynb` in Colab or Jupyter and run top to bottom. Update the paths at the top of the notebook if not using Google Drive.

## Reference

Pang, G., Shen, C., Jin, H., & van den Hengel, A. (2023). *Deep weakly-supervised anomaly detection.* KDD '23. ACM. [arXiv:1910.13601](https://arxiv.org/pdf/1910.13601)
