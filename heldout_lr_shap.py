#!/usr/bin/env python3
"""Held-out SHAP feature importance for the uncalibrated Logistic Regression.

Figure 5 accompanying the revised manuscript. Requires a PRIVATE local CSV
with one row per patient and columns study_id, label, and nlp_text OR
cleaned_preview. No clinical records are supplied with this repository.

This separate five-fold analysis does not change the primary 5x20 validation
or its operating thresholds. It explains the raw logistic regression margin
(log-odds), not the internally calibrated probabilities.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from bd_ct_nlp.review_analysis import make_tfidf, get_models


def calculate(input_csv: Path, output_dir: Path, seed: int = 42,
              expected_n: int | None = 161) -> pd.DataFrame:
    import shap
    data = pd.read_csv(input_csv)
    col = 'cleaned_preview' if 'cleaned_preview' in data else 'nlp_text'
    required = {'study_id','label',col}
    if not required.issubset(data.columns):
        raise ValueError('Expected columns: study_id, label, nlp_text (or cleaned_preview).')
    data=data[['study_id','label',col]].dropna().copy()
    if data.study_id.duplicated().any():
        raise ValueError('One row per patient required.')
    data=data.sort_values('study_id').reset_index(drop=True)
    if expected_n is not None and len(data)!=expected_n:
        raise ValueError(f'Expected {expected_n} patients, received {len(data)}')
    texts=data[col].astype(str).tolist()
    labels=data['label'].astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Exactly two outcome classes (0/1) are required.')
    if min(np.bincount(labels)) < 5:
        raise ValueError('Each outcome class must have >= 5 patients.')

    total_abs=pd.Series(dtype=float)
    top20_counts=Counter()
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    for fold,(train,test) in enumerate(cv.split(texts,labels),1):
        vectorizer=make_tfidf()
        x_train=vectorizer.fit_transform([texts[i] for i in train])
        x_test=vectorizer.transform([texts[i] for i in test])
        model=clone(get_models()['LR']).fit(x_train,labels[train])
        background=shap.maskers.Independent(x_train,max_samples=x_train.shape[0])
        explainer=shap.LinearExplainer(model,background)
        values=np.asarray(explainer(x_test).values)
        raw=model.decision_function(x_test)
        offset=float(np.asarray(explainer.expected_value).ravel()[0])
        np.testing.assert_allclose(offset+values.sum(axis=1),raw,atol=1e-5)
        features=vectorizer.get_feature_names_out()
        fold_sum=pd.Series(np.abs(values).sum(axis=0),index=features)
        total_abs=total_abs.add(fold_sum,fill_value=0)
        top20_counts.update(pd.Series(np.abs(values).mean(axis=0),
                                       index=features).nlargest(20).index.tolist())
        print(f'Fold {fold}: {len(test)} held-out patients; {len(features)} features')

    table=pd.DataFrame({'feature':total_abs.index,
                        'mean_abs_shap':total_abs.values / len(data)})
    table['top20_folds']=(table.feature.map(top20_counts).fillna(0).astype(int))
    table=table.sort_values('mean_abs_shap',ascending=False).head(20).reset_index(drop=True)
    output_dir.mkdir(parents=True,exist_ok=True)
    table.to_csv(output_dir/'LR_SHAP_top20.csv',index=False)

    fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,8),sharey=True,
                             gridspec_kw={'width_ratios':[4.3,1],'wspace':0.06})
    pos=np.arange(len(table))
    ax.barh(pos,table.mean_abs_shap,height=0.72)
    ax.set_yticks(pos,table.feature,fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean absolute SHAP value (log-odds)')
    ax.set_xlim(0,table.mean_abs_shap.max()*1.16)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=0.18); ax.set_axisbelow(True)
    ax2.scatter(table.top20_folds,pos,s=65)
    ax2.set_xlim(-0.3,5.5); ax2.set_xticks(range(6))
    ax2.set_xlabel('Top-20 folds (of 5)')
    ax2.tick_params(axis='y',left=False,labelleft=False)
    ax2.spines[['top','right','left']].set_visible(False)
    ax2.grid(axis='x',alpha=0.18); ax2.set_axisbelow(True)
    fig.suptitle('Logistic Regression — SHAP feature importance',fontsize=14,y=.99)
    fig.text(.5,.015,'Five-fold held-out analysis | Uncalibrated model | Raw log-odds output',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.24,right=.97,top=.94,bottom=.10)
    fig.savefig(output_dir/'Figure_5_LR_SHAP.png',dpi=400,bbox_inches='tight')
    fig.savefig(output_dir/'Figure_5_LR_SHAP.pdf',bbox_inches='tight')
    plt.close(fig)
    return table

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-csv',required=True,type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--expected-n',type=int,default=161)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    calculate(args.private_csv,args.out_dir,args.seed,args.expected_n)
