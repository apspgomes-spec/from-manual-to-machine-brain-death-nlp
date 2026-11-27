# From Manual to Machine: Development and Internal Validation of an NLP-Based Prediction Model for Brain Death Using Admission CT Reports

**Research software — release candidate for private review | Study cohort: 161 patients (94 BD, 67 NBD)**

Gomes A, França A, Sousa E, Gonçalves A, Fernandes AP. *Diagnostics* 2026;16:3995. https://doi.org/10.3390/diagnostics16233995.

This package contains the source code for the text-processing and machine-learning procedures described in the published article, together with synthetic examples and automated tests. It is intended to facilitate methodological assessment and research reuse; **no patient-level data are included**.

## Contents

- `src/bd_ct_nlp/additional_analyses.py` — documentary ablation and temporal validation; study-specific cohort checks are retained.

- `src/bd_ct_nlp/pipeline.py` — PDF text extraction, Portuguese-language normalization, administrative text filtering, and lexical negation checks.
- `src/bd_ct_nlp/review_analysis.py` and `review_cli.py` — cohort reconciliation, TF-IDF models, repeated stratified cross-validation, train-only sigmoid calibration and threshold selection, performance estimation and patient-level bootstrap.
- `src/bd_ct_nlp/metadata_controls.py` — exploratory year, document-template and report-length controls.
- `src/bd_ct_nlp/visualization.py` — descriptive report-length, word-cloud and n-gram visualizations.
- `examples/heldout_lr_shap.py` — separate five-fold held-out SHAP analysis of **uncalibrated Logistic Regression**, corresponding to **Figure 5**. SHAP values explain raw log-odds, not calibrated probabilities.
- `examples/demo_synthetic.py`, `examples/synthetic_*.txt`, `tests/` — synthetic examples and automated checks.

## Installation and test

Python 3.10 or newer is required. Run from the package directory:

```bash
python -m pip install -e ".[analysis,test]"
python -m pytest -q
python examples/demo_synthetic.py
```

## Running with the protected study data

The primary analysis requires the locally held clinical workbook, the index CT PDFs and the preprocessed report text export. All input files and generated patient-level outputs must remain on institutional, access-controlled storage, **outside this archive**.

```bash
python -m bd_ct_nlp.review_cli \
  --base-dir /PRIVATE/BD_AI \
  --clinical-file /PRIVATE/BDNBD_corrected.xlsx \
  --v5-csv /PRIVATE/PRIVATE_text_processing_full.csv \
  --out-dir /PRIVATE/reanalysis \
  --mode final --target 0.95 --fit-models --metadata-controls --wordclouds
```

The `--v5-csv` option is the historical **input file name** for the final preprocessed-text export, not a model name or a separate analysis. The source report extractor, if rerun from PDFs, can require an additional locally held clinician-footer marker file (not included for confidentiality). See `python -m bd_ct_nlp --help` for that preprocessing-only entry point.

The revised manuscript's **Figure 5** is produced by a separate held-out logistic-regression SHAP procedure:

```bash
python examples/heldout_lr_shap.py \
  --private-csv /PRIVATE/locked_patients.csv \
  --out-dir /PRIVATE/LR_SHAP
```

The private SHAP CSV requires one row per patient with `study_id`, `label` (1=BD, 0=NBD), and `nlp_text` (or `cleaned_preview`). The optional `--shap` argument on `review_cli.py` instead invokes an exploratory **XGBoost** SHAP analysis and **is not** the source of Figure 5; do not use it for the revised figure.

## Important limitations

- Clinical data, PDFs, patient identifiers, physician-footer marker lists, private audit files and individual predictions are intentionally excluded. Only synthetic examples are supplied.
- Automated tests verify aspects of code behaviour on synthetic inputs; **they cannot independently reproduce the numerical results reported in the manuscript**. Exact reproduction requires authorized access to the locked clinical cohort and site-specific extraction configuration.
- Repeated validation folds are not independent observations; uncertainty estimation preserves patient-level repeated predictions. The study is a retrospective, single-centre internal validation, not a prospective or external validation.
- Temporal case-selection imbalance and document-template differences may affect model discrimination. Exploratory controls should not be interpreted as proof of clinical validity.
- No patient data or outputs should be published or shared further without relevant institutional review and authorization.

**Note on figure numbering:** The separate logistic-regression SHAP script has been relabelled to Figure 5 to match the final manuscript; its computational method has not been changed. The original reviewer modules are retained. Additional analyses were ported from the final reviewer notebook.

## Additional analyses

The 100% training sensitivity target can be run using the primary CLI with `--target 1.0` and a separate output directory.

For documentary feature ablation and temporal analyses, first run the primary 95% analysis above, then run:

```bash
python -m bd_ct_nlp.additional_analyses --base-dir /PRIVATE/BD_AI --clinical-file /PRIVATE/BDNBD_corrected.xlsx --v5-csv /PRIVATE/PRIVATE_text_processing_full.csv --out-dir /PRIVATE/additional --primary-dir /PRIVATE/reanalysis/local_results_review_before_release
```

Temporal evaluation uses joint period/outcome stratification in the restricted pooled cohort, with period-specific metrics, and separate cross-period transfer. The ablation removes fitted n-gram features containing selected documentary tokens. All outputs must remain private until reviewed.

## Status and intended use

Research only; not a clinical diagnostic tool or a basis for decisions about treatment or organ donation. No external validation is established. The preprocessing implementation is not certified as byte-identical to the archived V5 text. Software authors (in article order): António Gomes, Ana França, Eduardo Sousa, António Gonçalves and Ana Paula Fernandes. Maintainer: apspgomes-spec. Licensed under the MIT License. See LICENSE. The software is prepared for release after final repository review.

## Verification status

The original 14 automated tests and synthetic preprocessing demo passed. The sensitivity/ablation and temporal/transfer routines completed an execution-only synthetic smoke test with reduced repetitions and bootstrap draws (two repetitions and five draws). The released analysis defaults remain unchanged (20 repetitions and 2000 bootstrap draws). The study-specific frozen V5 text, private configuration, and exact numerical results cannot be independently verified with the public files. The test environment used Python 3.13.5, scikit-learn 1.8.0, pandas 2.2.3, NumPy 2.3.5, SciPy 1.17.0, XGBoost 3.1.3, Matplotlib 3.10.8 and PyMuPDF 1.26.7; it differs from the environment reported in the article.
