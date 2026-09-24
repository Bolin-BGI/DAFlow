# DAFlow

Anonymous code release accompanying a double-blind submission.

DAFlow forecasts future cell-state distributions from time-resolved single-cell
RNA-seq snapshots. It combines three components in a frozen-autoencoder latent
space:

- a deterministic, horizon-conditioned displacement network (drift),
- a conditional residual flow that refines the stochastic component of the
  transition,
- a nonstationary, prototype-level growth field supervised with unbalanced
  optimal transport, which reallocates population mass.

Training uses direct multi-horizon supervision; inference mixes direct-horizon
experts with ancestor resampling under the learned growth field.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Place the datasets under `new_data/` (paths can be overridden with
`--data-root`; see `CONFIGS` in `model.py`):

| Dataset | Source | Expected location |
|---|---|---|
| Murine heart (CD1) | GEO: GSE193346 | `new_data/GSE193346_CD1_embryonic.h5ad` |
| In-vitro beta-cell differentiation | GEO: GSE114412 (protocol x1, stages S3c-S6c) | `new_data/vitrobetacell/h5ad_x1/` |
| MEF-to-iPSC reprogramming | Schiebinger et al. 2019 (WOT tutorial data) | `new_data/Schiebinger2019/reduce_processed/` |
| Zebrafish embryogenesis | Broad Single Cell Portal: SCP162 | `new_data/zebrafish_embryonic/new_processed/` |

## Usage

Train and evaluate on one dataset:

```bash
python model.py --dataset cd1 --seed 11
```

Datasets: `cd1`, `zebrafish`, `wot`, `wot_early`, `vitrobetacell`.
Useful flags: `--checkpoint` (inference from a saved checkpoint),
`--max-horizon`, `--horizon-weights`, `--ensemble-weights`,
`--mmd-weight`, `--consistency-weight`, `--bt-jitter`.

Each run writes per-time-point W2 reports (JSON), generated-cell `.h5ad`
files, and checkpoints under `<dataset>_protomass_clean_*` directories.
