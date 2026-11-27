"""Document-format and calendar-year negative controls for outcome prediction.

These predictors intentionally omit clinical report content. Fit on outer training
partitions only. Their performance assesses possible dataset construction and
reporting artefacts; it is not a substitute for an external/temporal validation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def metadata_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build non-radiological predictors from V5-linked cohort audit columns."""
    if 'ct_date' not in cohort or 'template' not in cohort:
        raise ValueError('ct_date and template are required for negative controls')
    x=cohort.copy()
    d=pd.to_datetime(x['ct_date'],errors='coerce')
    x['report_year']=d.dt.year.astype('Int64').astype('string').fillna('unknown').astype(str)
    x['report_format']=x['template'].fillna('unknown').astype(str)
    if 'clinical_words' in x.columns:
        length=pd.to_numeric(x['clinical_words'],errors='coerce')
    else:
        length=x['nlp_text'].fillna('').astype(str).str.split().str.len()
    x['log_report_length']=np.log1p(length.clip(lower=0))
    x['log_report_length']=x['log_report_length'].fillna(x['log_report_length'].median())
    if x['log_report_length'].isna().any():
        raise ValueError('No valid report lengths')
    return x


def control_estimator(name: str) -> Pipeline:
    choices={
        'Year_only':(['report_year'],[]),
        'Format_only':(['report_format'],[]),
        'Length_only':([],['log_report_length']),
        'Year_format':(['report_year','report_format'],[]),
        'All_metadata':(['report_year','report_format'],['log_report_length']),
    }
    if name not in choices:
        raise ValueError(f'Unknown control {name}')
    categorical,numeric=choices[name]
    transformers=[]
    if categorical:
        transformers.append(('cat',OneHotEncoder(handle_unknown='ignore'),categorical))
    if numeric:
        transformers.append(('length',Pipeline([
            ('impute',SimpleImputer(strategy='median')),
            ('scale',StandardScaler()),
        ]),numeric))
    return Pipeline([
        ('features',ColumnTransformer(transformers,remainder='drop')),
        ('model',LogisticRegression(max_iter=2000,class_weight='balanced')),
    ])


def run_negative_controls(cohort: pd.DataFrame, saved_nlp_oof: pd.DataFrame,
                          out_dir: Path, *, repeats:int=20, splits:int=5,
                          seed:int=42) -> pd.DataFrame:
    """Use exactly the same patient partitions as the historical NLP OOF file.

    Requires an OOF file generated with the same seed/repeats/splits. Fails closed
    on mismatches rather than silently comparing different splits.
    """
    x=metadata_features(cohort).reset_index(drop=True)
    if x['study_id'].duplicated().any():
        raise ValueError('Repeated patient ID in clinical data')
    y=x['label'].to_numpy(int)
    if len(np.unique(y))!=2:
        raise ValueError('Both outcome classes required')
    ref=saved_nlp_oof.loc[saved_nlp_oof['model']=='LR'].copy()
    if ref.empty:
        raise ValueError('Missing LR OOF reference predictions')
    check=ref.groupby('study_id').size()
    if set(check.index)!=set(x.study_id) or not check.eq(repeats).all():
        raise ValueError('NLP OOF count/IDs do not match cohort and expected repeats')
    out=[]; patient_predictions=[]
    cv=RepeatedStratifiedKFold(n_splits=splits,n_repeats=repeats,random_state=seed)
    model_names=['Year_only','Format_only','Length_only','Year_format','All_metadata']
    for k,(tr,te) in enumerate(cv.split(x,y)):
        rep,fold=divmod(k,splits)
        ids=x.iloc[te]['study_id'].tolist()
        locked=ref[(ref['repeat']==rep)&(ref['fold']==fold)].set_index('study_id')
        if set(locked.index)!=set(ids):
            raise ValueError(f'Partition mismatch in repetition {rep}, fold {fold}')
        out.append(dict(model='LR_NLP_reference',repeat=rep,fold=fold,
                        auc=roc_auc_score(y[te],locked.loc[ids,'probability'])))
        for name in model_names:
            mdl=control_estimator(name)
            mdl.fit(x.iloc[tr],y[tr])
            p=mdl.predict_proba(x.iloc[te])[:,1]
            out.append(dict(model=name,repeat=rep,fold=fold,
                            auc=roc_auc_score(y[te],p)))
            patient_predictions.extend(dict(repeat=rep,fold=fold,model=name,
                        study_id=sid,label=int(yy),probability=float(pp))
                        for sid,yy,pp in zip(ids,y[te],p))
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    fold_results=pd.DataFrame(out)
    fold_results.to_csv(out_dir/'metadata_negative_control_fold_auc.csv',index=False)
    # Patient level predictions are potentially sensitive; store privately.
    private=out_dir/'private_sensitive_do_not_share'
    private.mkdir(exist_ok=True)
    pd.DataFrame(patient_predictions).to_csv(private/'metadata_oof_PRIVATE.csv',index=False)
    return fold_results
