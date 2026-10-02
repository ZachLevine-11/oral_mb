# oral_mb

Oral-microbiome × phenome analysis pipeline (HPP buccal shotgun). No DL — LightGBM + classical stats only.

## Data

Set the input and output locations with environment variables:

```bash
export ORAL_MB_DATA_DIR=/path/to/paper_dfs   # default: data/paper_dfs
export ORAL_MB_OUT_DIR=/path/to/runs         # default: runs
```

Files: `{species,genus,family,phylum,pathways}.parquet`, `symptoms.parquet`,
`lifestyle.parquet`, `covariates.parquet`. MetaPhlAn4 log10 with absent
sentinel `-4.0`.

## Layout

```
oral_mb/
  config.py        paths + constants
  data.py          loaders, CLR transform, prevalence filter
  stats_utils.py   BH/Storey q-values, bootstrap CIs, calibration
  associations.py  per-feature regression + FDR + sex-stratified
  prediction.py    LightGBM CV, AUC/AUPRC/Brier/ECE w/ bootstrap CI
  mediation.py     exposure -> microbe -> outcome (Baron-Kenny + bootstrap)
  longitudinal.py  within-person fixed-effects for repeat-visit HPP
  mgwas.py         INT phenotypes, PLINK2/GCTA runners, coloc
  strain.py        StrainPhlAn/HUMAnN wrappers, virulence-gene tests
  replication.py   external-cohort replication + IVW meta-analysis
  sensitivity.py   BMI / sex robustness
  plots.py         heatmaps, calibration, forest plots
scripts/
  run_pipeline.py     discovery + sensitivity + prediction + mediation + plots
  run_mgwas_prep.py   write PLINK/GCTA phenotype + covariate files
  run_replication.py  replicate + meta-analyze across cohorts
tests/
  test_smoke.py       offline smoke tests on synthetic cohort
```

## Setup

```bash
conda create -n microbiome python=3.11 -y
conda activate microbiome
pip install -e '.[test]'
pytest -q
```

## Run

```bash
# Full pipeline, all levels
python -m scripts.run_pipeline --mediate \
  --out ./runs

# mGWAS prep
python -m scripts.run_mgwas_prep --level species --bfile /path/HPP_geno

# External replication
python -m scripts.run_replication \
  --discovery-dir /path/to/paper_dfs \
  --replication-dir /path/external/paper_dfs \
  --level species --out ./rep_out
```

## Outputs per level

```
<out>/<level>/
  associations_{all,symptoms,lifestyle}.csv
  shared_signature.csv  heatmap_shared.png
  sensitivity_*.csv
  prediction_lgbm.csv  auc_bars.png  pearson_bars.png
  calibration_bleeding_gums.png
  mediation.csv  (if --mediate)
```
