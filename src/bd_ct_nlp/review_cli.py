"""Command-line entry point for controlled-access study reanalysis."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from . import review_analysis as analysis


def main(argv=None):
    ap=argparse.ArgumentParser(description='Reproduce the BD CT-NLP reviewer analysis locally.')
    ap.add_argument('--base-dir',required=True,type=Path,help='PRIVATE directory with BD/ and NBD/ PDF folders')
    ap.add_argument('--clinical-file',required=True,type=Path,help='PRIVATE corrected clinical workbook')
    ap.add_argument('--v5-csv',required=True,type=Path,help='PRIVATE frozen V5 output CSV')
    ap.add_argument('--out-dir',required=True,type=Path,help='PRIVATE output directory; not inside Git repository')
    ap.add_argument('--mode',choices=('test','final'),default='test')
    ap.add_argument('--fit-models',action='store_true',help='Run repeated CV; otherwise only audit/reconcile')
    ap.add_argument('--target',type=float,default=.95,help='Target sensitivity selected within outer-train (0,1]')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--repeats',type=int)
    ap.add_argument('--bootstrap',type=int)
    ap.add_argument('--shap',action='store_true',help='Experimental: XGBoost fold-level raw-output SHAP')
    ap.add_argument('--metadata-controls',action='store_true',help='Year/template/length negative control on same held-out splits')
    ap.add_argument('--wordclouds',action='store_true',help='Six-panel figure and supplementary 2/3-gram clouds')
    args=ap.parse_args(argv)
    if args.metadata_controls and not args.fit_models:
        ap.error('--metadata-controls requires --fit-models in the same run')
    analysis.configure(base_dir=args.base_dir,clinical_file=args.clinical_file,
                       v5_csv=args.v5_csv,out_dir=args.out_dir,
                       mode=args.mode,target=args.target,seed=args.seed,
                       repeats=args.repeats,bootstrap=args.bootstrap)
    result=analysis.execute(fit_models=args.fit_models,run_shap=args.shap and args.fit_models)
    if args.metadata_controls:
        from .metadata_controls import run_negative_controls
        neg=run_negative_controls(result['data'],result['oof'],analysis.OUT_MANUSCRIPT,
                                  repeats=analysis.N_REPEATS,splits=analysis.N_SPLITS,seed=analysis.SEED)
        print('Negative-control mean AUC by fold (descriptive):')
        print(neg.groupby('model')['auc'].agg(['mean','std']).round(4))
    if args.wordclouds:
        from .visualization import render_wordclouds
        summary=render_wordclouds(args.v5_csv,analysis.OUT_PRIVATE/'figures_pending_privacy_review',
                                  expected_bd=int(result['data'].label.sum()),
                                  expected_nbd=int((result['data'].label==0).sum()))
        print('Descriptive figures need manual screening for patient identifiers:')
        print(json.dumps(summary,indent=2))
    print('COMPLETE; retain all patient-level output in controlled-access storage.')

if __name__=='__main__':
    main()
