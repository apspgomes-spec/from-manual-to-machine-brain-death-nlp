"""Clinical cohort reconciliation and repeated stratified validation.

Adapted from the study's internally executed reviewer-analysis workflow.
This module contains no patient data. All inputs remain local and confidential.
"""
from __future__ import annotations

from pathlib import Path
import re, sys, json, hashlib, unicodedata, warnings, platform
import importlib.metadata as md
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, brier_score_loss, confusion_matrix,
                             accuracy_score, f1_score, log_loss, roc_curve)
from xgboost import XGBClassifier

BASE_DIR = None
CLINICAL_FILE = None
V5_CSV = None
OUTPUT = None
OUT_PRIVATE = None
OUT_MANUSCRIPT = None
MODE = 'test'
N_SPLITS = 5
N_REPEATS = 2
N_BOOT = 100
SEED = 42
SENSITIVITY_TARGET = 0.95
USE_CURATED_WHITELIST = False
CURATED_CSV = None
RUN_SHAP = False
RUN_COMBINED_CLINICAL_NLP = False
RUN_MODEL_TRAINING = False


def configure(*, base_dir: Path, clinical_file: Path, v5_csv: Path,
              out_dir: Path, mode: str = 'test', target: float = 0.95,
              seed: int = 42, bootstrap: int | None = None,
              repeats: int | None = None, combine_clinical_nlp: bool = False) -> None:
    """Set an isolated analysis run configuration; never overwrite another target."""
    global BASE_DIR, CLINICAL_FILE, V5_CSV, OUTPUT, OUT_PRIVATE, OUT_MANUSCRIPT
    global MODE, N_REPEATS, N_BOOT, SEED, SENSITIVITY_TARGET
    global USE_CURATED_WHITELIST, CURATED_CSV, RUN_COMBINED_CLINICAL_NLP
    if mode not in ('test','final'):
        raise ValueError('mode must be test or final')
    if not 0 < target <= 1:
        raise ValueError('target must be within (0, 1]')
    if repeats is not None and repeats < 1:
        raise ValueError('repeats must be >= 1')
    BASE_DIR = Path(base_dir).expanduser().resolve()
    CLINICAL_FILE = Path(clinical_file).expanduser().resolve()
    V5_CSV = Path(v5_csv).expanduser().resolve()
    OUTPUT = Path(out_dir).expanduser().resolve()
    OUT_PRIVATE = OUTPUT / 'private_sensitive_do_not_share'
    OUT_MANUSCRIPT = OUTPUT / 'local_results_review_before_release'
    for p in (OUTPUT, OUT_PRIVATE, OUT_MANUSCRIPT): p.mkdir(parents=True, exist_ok=True)
    MODE = mode
    N_REPEATS = int(repeats if repeats is not None else (20 if mode == 'final' else 2))
    N_BOOT = int(bootstrap if bootstrap is not None else (2000 if mode == 'final' else 100))
    SEED = int(seed)
    SENSITIVITY_TARGET = float(target)
    USE_CURATED_WHITELIST = False
    CURATED_CSV = None
    RUN_COMBINED_CLINICAL_NLP = bool(combine_clinical_nlp)




def normalized_key(value):
    if pd.isna(value): return None
    txt = str(value).strip()
    if txt.lower().endswith('.pdf'): txt = txt[:-4]
    if re.fullmatch(r'\d+\.0', txt): txt = txt[:-2]
    return txt


def pdf_inventory(base):
    rec = []
    for folder, y in [('BD', 1), ('NBD', 0)]:
        path = base / folder
        if not path.is_dir():
            raise FileNotFoundError(f'Missing source folder: {path}')
        for f in sorted(path.glob('*.pdf'), key=lambda p: p.name.casefold()):
            rec.append(dict(process=normalized_key(f.stem), path=str(f), pdf_name=f.name,
                            group_pdf=folder, label_pdf=y))
    files = pd.DataFrame(rec)
    if files.empty: raise ValueError('No source PDFs found.')
    if files['process'].isna().any() or files['process'].duplicated().any():
        raise ValueError('Missing/duplicated process IDs among PDF files; inspect folder inventory.')
    return files


def standardize_clinical(raw):
    c = raw.copy()
    c.columns = [str(x).strip().lower() for x in c.columns]
    rename = {'processo':'process', 'bd/nbd':'group_clinical', 'idade':'age',
              'sexo':'sex', 'diag':'etiology', 'dt_tc':'ct_date',
              'dt_bd':'outcome_date', 'alta/obito':'outcome_type'}
    missing = [k for k in rename if k not in c.columns]
    if missing: raise ValueError(f'Clinical workbook missing columns: {missing}; found: {c.columns.tolist()}')
    c = c.rename(columns=rename)[list(rename.values())]
    c['process'] = c['process'].map(normalized_key)
    c['group_clinical'] = c['group_clinical'].astype('string').str.strip().str.upper()
    c['label_clinical'] = c['group_clinical'].map({'BD':1,'NBD':0})
    c['age'] = pd.to_numeric(c['age'], errors='coerce')
    c['sex'] = c['sex'].astype('string').str.strip().replace({'<NA>':pd.NA})
    c['etiology'] = c['etiology'].astype('string').str.strip().str.upper()
    normalize_dx = {'ISCHEMIC STROKE':'IS', 'ISCHAEMIC STROKE':'IS',
                    'AVC ISQUEMICO':'IS', 'HSA':'SAH', 'HSD':'SDH'}
    c['etiology'] = c['etiology'].replace(normalize_dx)
    for date in ('ct_date', 'outcome_date'):
        c[date] = pd.to_datetime(c[date], errors='coerce', dayfirst=True)
    c['outcome_type'] = c['outcome_type'].astype('string').str.strip().str.lower()
    return c


def reconcile_source(base, clinical_file):
    pdf = pdf_inventory(base)
    if not clinical_file.exists(): raise FileNotFoundError(f'Missing: {clinical_file}')
    clinical = standardize_clinical(pd.read_excel(clinical_file))
    if clinical['process'].isna().any() or clinical['process'].duplicated().any():
        raise ValueError('Clinical data have empty or duplicate process IDs')
    joined = pdf.merge(clinical, how='outer', on='process', validate='one_to_one', indicator=True)
    joined['issue'] = ''
    joined.loc[joined['_merge']!='both', 'issue'] = 'MISSING_PDF_OR_CLINICAL'
    mismatch = joined['_merge'].eq('both') & joined['label_pdf'].ne(joined['label_clinical'])
    joined.loc[mismatch, 'issue'] = 'LABEL_MISMATCH'
    if joined['label_clinical'].isna().any():
        joined.loc[joined['label_clinical'].isna(), 'issue'] = 'UNKNOWN_CLINICAL_LABEL'
    joined.to_csv(OUT_PRIVATE/'source_reconciliation_PRIVATE.csv', index=False)
    problems = joined[joined['issue'].ne('')]
    if len(problems):
        print(problems[['process','pdf_name','group_pdf','group_clinical','issue']].to_string(index=False))
        raise ValueError('STOP: reconcile source files and clinical cohort BEFORE fitting any models.')
    joined = joined.sort_values('process').reset_index(drop=True)
    joined['study_id'] = ['P%04d' % (i+1) for i in range(len(joined))]
    joined['label'] = joined['label_pdf'].astype(int)
    joined['followup_days'] = (joined['outcome_date']-joined['ct_date']).dt.days
    joined[['study_id','process','pdf_name','label']].to_csv(OUT_PRIVATE/'patient_mapping_PRIVATE.csv', index=False)
    assert joined['study_id'].is_unique
    if (joined['followup_days'].dropna()<0).any():
        raise ValueError('Negative CT→outcome intervals; see private reconciliation file')
    if joined['age'].dropna().lt(18).any():
        raise ValueError('Confirmed patients younger than 18; cohort inclusion must be reviewed')
    outside_year = joined['ct_date'].notna() & ~joined['ct_date'].dt.year.between(2011,2024)
    if outside_year.any():
        raise ValueError('Index CT dates outside the declared 2011–2024 study period; check records')
    print('Reconciled cohort:', len(joined), '| BD',int(joined.label.sum()), '| NBD',int((joined.label==0).sum()))
    return joined

def read_pdfs(cohort):
    """Load frozen local V5 text; name retained to match the original ML notebook API.

    This function deliberately does NOT open or re-clean PDFs. It reconciles the
    V5 outputs against the clinical/PDF file inventory by filename and group.
    """
    if not V5_CSV.is_file():
        raise FileNotFoundError(
            f'Missing PRIVATE V5 text file: {V5_CSV}. Run/export the V5 NLP notebook first.'
        )
    raw = pd.read_csv(V5_CSV, encoding='utf-8-sig', dtype={'filename':str, 'group':str})
    required = {'filename','group','nlp_text','template','label'}
    if missing := (required-set(raw.columns)):
        raise ValueError(f'V5 output missing required columns: {sorted(missing)}')
    raw['pdf_name'] = raw['filename'].astype(str).str.strip()
    raw['group_v5'] = raw['group'].astype(str).str.strip().str.upper()
    raw['label_v5'] = pd.to_numeric(raw['label'], errors='raise').astype(int)
    if raw['pdf_name'].duplicated().any():
        raise ValueError('Duplicate filenames in V5 input — STOP.')
    if len(raw)!=161 or raw['group_v5'].value_counts().to_dict()!={'BD':94,'NBD':67}:
        raise ValueError('V5 cohort count mismatch (expected 94 BD / 67 NBD) — STOP.')
    if raw['nlp_text'].isna().any() or raw['nlp_text'].astype(str).str.strip().eq('').any():
        raise ValueError('V5 contains empty NLP documents — STOP.')
    if raw['nlp_text'].astype(str).str.contains(r'\b(?:neg|incerto)_[a-z]',regex=True).any():
        raise ValueError('Found neg_/incerto_ tokens in nlp_text. This is not the locked V5 primary representation.')
    if 'review_flags' in raw:
        unresolved = raw['review_flags'].fillna('').astype(str).str.contains(
            r'NO_ANCHOR|ADMIN_RESIDUE',regex=True
        )
        if unresolved.any():
            raise ValueError(f'V5 contains {int(unresolved.sum())} extraction/admin failures — STOP.')
        print('V5: short narrative flags:',int(raw['review_flags'].fillna('').str.contains('SHORT_CLINICAL_TEXT').sum()),
              '(to be clinically audited, not automatically excluded)')
    cols=['pdf_name','group_v5','label_v5','nlp_text','template']
    if 'review_flags' in raw:cols+=['review_flags']
    if 'clinical_words' in raw:cols+=['clinical_words']
    joined = cohort.drop(columns=['_merge'], errors='ignore').merge(raw[cols],on='pdf_name',how='outer',validate='one_to_one',indicator=True)
    if joined['_merge'].ne('both').any():
        raise ValueError('PDF/clinical inventory and V5 text filenames disagree — STOP.')
    if joined['group_pdf'].ne(joined['group_v5']).any() or joined['label'].ne(joined['label_v5']).any():
        raise ValueError('PDF/clinical groups and V5 labels disagree — STOP.')
    joined=joined.sort_values('study_id').drop(columns=['_merge']).reset_index(drop=True)
    joined['cleaned_preview']=joined['nlp_text'].astype(str) # NO fitting, NO preprocessing here
    print('Locked V5 successfully reconciled:',len(joined),'| BD',int(joined.label.sum()),
          '| NBD',int((joined.label==0).sum()))
    print('V5 extraction modes:',joined['template'].value_counts().to_dict())
    return joined


def make_tfidf():
    """Unfitted, label-free 1-3-gram TF-IDF, fitted inside each training fold."""
    return TfidfVectorizer(strip_accents='unicode', lowercase=False,
                           ngram_range=(1,3), token_pattern=r'(?u)\b\w+\b',
                           sublinear_tf=True, norm='l2', min_df=2, max_features=5000)


def get_models():
    return {
        'LR': LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', class_weight='balanced', random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                     class_weight='balanced_subsample', n_jobs=1, random_state=SEED),
        'SVM': SVC(C=1.0, kernel='rbf', gamma='scale', probability=False,
                   class_weight='balanced',random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                 reg_alpha=0.0,objective='binary:logistic',
                                 eval_metric='logloss', n_jobs=1,random_state=SEED),
    }


def prepare_estimator(model):
    pipe=Pipeline([('vectorizer',make_tfidf()),('classifier',clone(model))])
    return CalibratedClassifierCV(estimator=pipe, cv=3, method='sigmoid',ensemble=False)


def choose_threshold(y,p,target=0.95):
    # Predict positive when probability >= threshold.
    y=np.asarray(y); p=np.asarray(p)
    candidates=np.r_[np.nextafter(p.min(), -np.inf),np.unique(p)]
    viable=[]
    for t in candidates:
        pred=p>=t
        sens=np.mean(pred[y==1]); spec=np.mean(~pred[y==0])
        if sens>=target-1e-12: viable.append((spec,t))
    if not viable: raise ValueError('No threshold meets target; check outcome classes')
    return float(max(viable,key=lambda z:(z[0],z[1]))[1])


def train_one_outer_fold(X_train,y_train,X_test,base_model,seed):
    # Inner threshold selection: out-of-fold calibrated scores within OUTER TRAIN only.
    inner=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    base=prepare_estimator(base_model)
    train_oof=cross_val_predict(base,X_train,y_train,cv=inner,method='predict_proba',n_jobs=1)[:,1]
    threshold=choose_threshold(y_train,train_oof,SENSITIVITY_TARGET)
    fit=prepare_estimator(base_model)
    fit.fit(X_train,y_train)
    fitted_vectorizer=fit.calibrated_classifiers_[0].estimator.named_steps['vectorizer']
    n_features=len(fitted_vectorizer.get_feature_names_out())
    return fit.predict_proba(X_test)[:,1],threshold,n_features


def make_clinical_estimator(combined=False):
    # Encoded sex 1/2 is used as categorical, NOT assigned meaning without dictionary.
    features=['age','sex','etiology']
    steps=[('num',Pipeline([('fill',SimpleImputer(strategy='median')),
                            ('scale',StandardScaler())]),['age']),
           ('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),
                            ('encode',OneHotEncoder(handle_unknown='ignore'))]),['sex','etiology'])]
    if combined: steps.append(('nlp',make_tfidf(),'cleaned_preview'))
    pre=ColumnTransformer(steps, remainder='drop')
    base=Pipeline([('features',pre),('model',LogisticRegression(max_iter=2000,random_state=SEED))])
    return CalibratedClassifierCV(estimator=base,cv=3,method='sigmoid',ensemble=False)


def train_clinical_one_outer_fold(X_train,y_train,X_test,seed,combined=False):
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    base=make_clinical_estimator(combined)
    p_inner=cross_val_predict(base,X_train,y_train,cv=inner,method='predict_proba',n_jobs=1)[:,1]
    t=choose_threshold(y_train,p_inner,SENSITIVITY_TARGET)
    fit=make_clinical_estimator(combined)
    fit.fit(X_train,y_train)
    return fit.predict_proba(X_test)[:,1],t


def repeated_cv_predictions(data):
    labels=data['label'].to_numpy(dtype=int)
    text=data['cleaned_preview'].reset_index(drop=True)
    clinical=data[['age','sex','etiology','cleaned_preview']].reset_index(drop=True).copy()
    clinical['sex']=clinical['sex'].astype(object).replace({pd.NA:np.nan})
    clinical['etiology']=clinical['etiology'].astype(object).replace({pd.NA:np.nan})
    schema=['repeat','fold','model','study_id','label','probability','threshold','prediction']
    rows=[]; fold_rows=[]
    splitter=RepeatedStratifiedKFold(n_splits=N_SPLITS,n_repeats=N_REPEATS,random_state=SEED)
    models=get_models()
    for k,(train,test) in enumerate(splitter.split(text,labels)):
        rep,fold=divmod(k,N_SPLITS)
        ytr,yte=labels[train],labels[test]
        active=list(models)
        active+=['Clinical_structured']
        if RUN_COMBINED_CLINICAL_NLP: active+=['Clinical_plus_NLP']
        for name in active:
            print(f'CV {rep+1}/{N_REPEATS}, fold {fold+1}/{N_SPLITS}: {name}',flush=True)
            if name in models:
                prob,t,n_features=train_one_outer_fold(text.iloc[train],ytr,text.iloc[test],models[name],SEED+100*k)
            else:
                prob,t=train_clinical_one_outer_fold(clinical.iloc[train],ytr,
                                  clinical.iloc[test],SEED+100*k,name=='Clinical_plus_NLP')
                n_features=np.nan
            preds=(prob>=t).astype(int)
            for idx,p,yp in zip(test,prob,preds):
                rows.append((rep,fold,name,data.iloc[idx].study_id,int(labels[idx]),float(p),float(t),int(yp)))
            fold_rows.append(dict(repeat=rep,fold=fold,model=name,auc=roc_auc_score(yte,prob),
                                 sens=float(preds[yte==1].mean()),
                                 spec=float((1-preds[yte==0]).mean()),threshold=t,
                                 n=len(test),events=int(yte.sum()),n_tfidf_features=n_features))
    out=pd.DataFrame(rows,columns=schema)
    per_fold=pd.DataFrame(fold_rows)
    count=out.groupby(['model','study_id']).size()
    if not (count==N_REPEATS).all(): raise AssertionError('Not exactly one OOF per patient / repetition')
    out.to_csv(OUT_MANUSCRIPT/'oof_predictions_pseudonymous.csv',index=False)
    per_fold.to_csv(OUT_MANUSCRIPT/'fold_metrics_DESCRIPTIVE.csv',index=False)
    return out,per_fold

def metric_array(y,p,pred):
    if len(np.unique(y))<2: return None
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(auc=roc_auc_score(y,p), sensitivity=tp/(tp+fn),
                specificity=tn/(tn+fp), ppv=tp/(tp+fp) if tp+fp else np.nan,
                npv=tn/(tn+fn) if tn+fn else np.nan,
                accuracy=accuracy_score(y,pred), f1=f1_score(y,pred,zero_division=0),
                brier=brier_score_loss(y,p))


def build_prediction_cube(oof):
    ids=sorted(oof.study_id.unique())
    models=sorted(oof.model.unique())
    reps=sorted(oof['repeat'].unique())
    y=oof.drop_duplicates('study_id').set_index('study_id').loc[ids,'label'].to_numpy(int)
    cubes={}
    for model in models:
        subset=oof[oof.model==model]
        p=subset.pivot(index='repeat',columns='study_id',values='probability').loc[reps,ids].to_numpy(float)
        z=subset.pivot(index='repeat',columns='study_id',values='prediction').loc[reps,ids].to_numpy(int)
        if np.isnan(p).any(): raise ValueError('Missing prediction in cube')
        cubes[model] = (p,z)
    return ids,y,cubes


def summarise_repetitions(y,probs,decisions,idx=None):
    if idx is None: idx=np.arange(len(y))
    yy=y[idx]; results=[]
    for rep in range(len(probs)):
        m=metric_array(yy,probs[rep,idx],decisions[rep,idx])
        if m is not None: results.append(m)
    if not results: return None
    return {col:float(np.nanmean([d[col] for d in results])) for col in results[0]}


def patient_cluster_bootstrap(oof, B=N_BOOT):
    ids,y,cubes=build_prediction_cube(oof)
    rng=np.random.default_rng(SEED+51)
    central={model:summarise_repetitions(y,*cube) for model,cube in cubes.items()}
    draws={name:{m:[] for m in central[name]} for name in cubes}
    deltas={}
    # Patient indices sampled ONCE per bootstrap for EVERY model and repetition.
    # Thus paired between-model comparisons retain within-patient covariance.
    for b in range(B):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))!=2: continue
        for model,(p,decision) in cubes.items():
            m=summarise_repetitions(y,p,decision,ix)
            if m is None: continue
            for key,val in m.items(): draws[model][key].append(val)
    records=[]
    for model,vals in draws.items():
        for metric,dist in vals.items():
            dist=np.asarray(dist)
            records.append(dict(model=model,metric=metric,estimate=central[model][metric],
                                ci_lower=float(np.nanpercentile(dist,2.5)),
                                ci_upper=float(np.nanpercentile(dist,97.5)),n_boot=len(dist)))
    out=pd.DataFrame(records)
    out.to_csv(OUT_MANUSCRIPT/'patient_cluster_bootstrap_performance.csv',index=False)
    # AUC differences with *same* bootstrap draws for each model.
    for m in cubes:
        if m=='XGBoost' or 'XGBoost' not in cubes: continue
        x=np.asarray(draws['XGBoost']['auc']); z=np.asarray(draws[m]['auc'])
        if len(x)!=len(z): raise RuntimeError('Unpaired bootstrap arrays')
        d=x-z
        deltas[m]=dict(comparison='XGBoost - '+m,
                       difference=central['XGBoost']['auc']-central[m]['auc'],
                       ci_lower=float(np.percentile(d,2.5)),
                       ci_upper=float(np.percentile(d,97.5)))
    diffs=pd.DataFrame(deltas.values())
    diffs.to_csv(OUT_MANUSCRIPT/'paired_AUC_differences.csv',index=False)
    return out,diffs

def calibration_parameters(y,prob):
    y=np.asarray(y,int); prob=np.clip(np.asarray(prob,float),1e-6,1-1e-6)
    x=logit(prob)
    def nll_intercept(a):
        z=a[0]+x
        return np.logaddexp(0,z).sum()-(y*z).sum()
    def nll_joint(theta):
        z=theta[0]+theta[1]*x
        return np.logaddexp(0,z).sum()-(y*z).sum()
    a=minimize(nll_intercept,[0.0],method='BFGS')
    b=minimize(nll_joint,[0.0,1.0],method='BFGS')
    return float(a.x[0]),float(b.x[1]),float(b.x[0])


def calibration_and_threshold_audit(oof):
    rows=[]; by_patient=[]
    for model,sub in oof.groupby('model'):
        avg=sub.groupby('study_id').agg(label=('label','first'),p=('probability','mean')).sort_index()
        y=avg.label.to_numpy(int); p=avg.p.to_numpy(float)
        intercept,slope,joint_intercept=calibration_parameters(y,p)
        thresholds=sub.groupby(['repeat','fold'])['threshold'].first()
        rows.append(dict(model=model,auc_on_mean_patient_probability=roc_auc_score(y,p),
                         brier_on_mean_patient_probability=brier_score_loss(y,p),
                         calibration_intercept_offset=intercept,
                         calibration_slope=slope,
                         intercept_from_joint_calibration_regression=joint_intercept,
                         threshold_median=float(thresholds.median()),
                         threshold_q25=float(thresholds.quantile(.25)),
                         threshold_q75=float(thresholds.quantile(.75))))
        for row in avg.reset_index().itertuples(index=False):
            by_patient.append(dict(study_id=row.study_id,model=model,label=row.label,mean_probability=row.p))
    out=pd.DataFrame(rows)
    out.to_csv(OUT_MANUSCRIPT/'calibration_and_threshold_audit.csv',index=False)
    pd.DataFrame(by_patient).to_csv(OUT_MANUSCRIPT/'patient_mean_probabilities.csv',index=False)
    return out


def alert_scenarios(performance):
    rows=[]
    for model in sorted(performance.model.unique()):
        sub=performance[performance.model==model].set_index('metric')
        sens=float(sub.loc['sensitivity','estimate'])
        spec=float(sub.loc['specificity','estimate'])
        for prevalence in [0.01,0.05,0.10,0.20,0.50]:
            tp=100*prevalence*sens
            fp=100*(1-prevalence)*(1-spec)
            rows.append(dict(model=model,assumed_BD_prevalence=prevalence,
                true_alerts_per_100=tp,false_alerts_per_100=fp,
                total_alerts_per_100=tp+fp,
                clinical_reviews_per_true_case=(tp+fp)/tp if tp else np.nan,
                scenario_only='NOT an observed hospital prevalence'))
    result=pd.DataFrame(rows)
    result.to_csv(OUT_MANUSCRIPT/'alerts_scenarios_per100.csv',index=False)
    return result


def export_error_review(oof,clinical):
    base=clinical[['study_id','label','age','sex','etiology']].copy()
    sub=oof.groupby(['study_id','model']).agg(fp_rate=('prediction',lambda x:np.nan),
                                                  n=('prediction','size')).reset_index()
    working=oof.copy()
    working['false_positive']=((working.label==0)&(working.prediction==1)).astype(float)
    working['false_negative']=((working.label==1)&(working.prediction==0)).astype(float)
    err=working.groupby(['study_id','model']).agg(
        fp_fraction=('false_positive','mean'),fn_fraction=('false_negative','mean'),
        mean_probability=('probability','mean'),n_oof=('probability','size')).reset_index()
    err=err.merge(base,on='study_id',how='left',validate='many_to_one')
    err=err.sort_values(['model','fp_fraction','fn_fraction'],ascending=[True,False,False])
    err['physician_review']=''
    err['explanation_of_error']=''
    err.to_csv(OUT_PRIVATE/'FP_FN_clinician_review_PRIVATE.csv',index=False)
    return err

def clinical_descriptives(data):
    summary=[]
    for group,g in data.groupby('group_clinical'):
        ages=g['age'].dropna()
        summary.append(dict(group=group,n=len(g),n_age=ages.size,
            age_median=ages.median(), age_q25=ages.quantile(.25),age_q75=ages.quantile(.75),
            missing_age=int(g['age'].isna().sum()),missing_ct=int(g.ct_date.isna().sum()),
            missing_outcome=int(g.outcome_date.isna().sum()),
            followup_n=int(g.followup_days.notna().sum()),
            followup_median_days=g.followup_days.median(),
            followup_q25_days=g.followup_days.quantile(.25),
            followup_q75_days=g.followup_days.quantile(.75)))
    overall=pd.DataFrame(summary)
    overall.to_csv(OUT_MANUSCRIPT/'clinical_baseline_summary.csv',index=False)
    categorical=[]
    for col in ['sex','etiology','outcome_type']:
        for (group,value),n in data.groupby(['group_clinical',col],dropna=False).size().items():
            categorical.append(dict(variable=col,group=group,value=str(value),n=int(n)))
    pd.DataFrame(categorical).to_csv(OUT_MANUSCRIPT/'clinical_baseline_categories.csv',index=False)
    by_year=data.groupby(['group_clinical',data.ct_date.dt.year],dropna=False).size().rename('n').reset_index()
    by_year.to_csv(OUT_MANUSCRIPT/'clinical_selection_by_year.csv',index=False)
    return overall



def feature_space_report(data, fold_metrics):
    """Describe per-fold vocabulary sizes and the full-data REFIT vocabulary.

    The full-data refit is for future model deployment and descriptive reporting,
    NEVER for preselection of validation features.
    """
    full=make_tfidf()
    full.fit(data['cleaned_preview'])
    features=full.get_feature_names_out()
    sizes=fold_metrics.groupby('model')['n_tfidf_features'].agg(['min','median','max']).reset_index()
    sizes['full_data_refit_feature_count']=np.where(sizes['model'].isin(get_models()),len(features),np.nan)
    sizes.to_csv(OUT_MANUSCRIPT/'tfidf_feature_counts.csv',index=False)
    # Terms may reflect original patient reports; retain in controlled access only.
    pd.DataFrame({'term':features}).to_csv(OUT_PRIVATE/'tfidf_full_refit_terms_PRIVATE.csv',index=False)
    return sizes


def summarise_manual_negation_accuracy():
    filepath=OUT_PRIVATE/'negation_manual_validation_PRIVATE.csv'
    if not filepath.exists(): return None
    review=pd.read_csv(filepath)
    if 'reviewer_correct' not in review: return None
    ans=review['reviewer_correct'].astype('string').str.strip().str.lower()
    valid=ans.isin(['yes','no','sim','nao','não','correct','incorrect'])
    if not valid.any():
        print('Negation QC: clinician review PENDING; no accuracy claimed')
        return None
    ok=ans.isin(['yes','sim','correct'])
    summary=dict(n_reviewed=int(valid.sum()),n_correct=int((ok & valid).sum()),
                 accuracy=float((ok & valid).sum()/valid.sum()),
                 caution='Only reflects the manually adjudicated sample')
    pd.DataFrame([summary]).to_csv(OUT_MANUSCRIPT/'negation_manual_QC_summary.csv',index=False)
    return summary

def shap_stability_exploratory(data):
    try:
        import shap
    except ImportError:
        warnings.warn('Install shap to run explainability.'); return
    labels=data['label'].to_numpy(int); texts=data['cleaned_preview'].tolist()
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    feature_importances=[]
    for fold,(train,test) in enumerate(cv.split(texts,labels)):
        v=make_tfidf()
        Xtr=v.fit_transform([texts[i] for i in train]); Xte=v.transform([texts[i] for i in test])
        mdl=clone(get_models()['XGBoost']).fit(Xtr,labels[train])
        # TreeExplainer explains RAW XGBoost log-odds (not Platt-calibrated probability).
        explainer=shap.TreeExplainer(mdl,model_output='raw')
        values=explainer.shap_values(Xte.toarray())
        if isinstance(values,list): values=values[-1]
        if np.ndim(values)==3: values=np.asarray(values)[:,:,1]
        importance=np.mean(np.abs(values),axis=0)
        s=pd.Series(importance,index=v.get_feature_names_out(),name=f'fold{fold+1}')
        feature_importances.append(s)
    matrix=pd.concat(feature_importances,axis=1).fillna(0.0)
    matrix['mean_abs_SHAP']=matrix.mean(axis=1)
    matrix.sort_values('mean_abs_SHAP',ascending=False).to_csv(OUT_MANUSCRIPT/'SHAP_fold_importance_RAW.csv')
    # Focus on the union of each fold's top 25 terms to avoid inflated
    # correlations from thousands of features with tied zero importance.
    ranks=matrix.drop(columns='mean_abs_SHAP')
    top_union=set().union(*(set(ranks[col].nlargest(25).index) for col in ranks.columns))
    corr=ranks.loc[sorted(top_union)].rank().corr(method='spearman')
    corr.to_csv(OUT_MANUSCRIPT/'SHAP_rank_stability_spearman.csv')
    print('SHAP fold-rank correlations:'); print(corr.round(3))

def figures(oof,perf,calibration):
    f,ax=plt.subplots(figsize=(6.4,5.2))
    for model,g in oof.groupby('model'):
        # Compare across unique patients (20 predictions averaged per patient)
        a=g.groupby('study_id').agg(y=('label','first'),p=('probability','mean'))
        fpr,tpr,_=roc_curve(a.y,a.p)
        ax.plot(fpr,tpr,label=f'{model} (AUC={roc_auc_score(a.y,a.p):.3f})')
    ax.plot([0,1],[0,1],ls='--',c='0.6')
    ax.set(xlabel='1 − specificity',ylabel='Sensitivity',title='Mean-patient OOF probabilities')
    ax.legend(fontsize=8,loc='lower right');f.tight_layout()
    f.savefig(OUT_MANUSCRIPT/'ROC_patient_mean_probabilities.png',dpi=220);plt.close(f)
    f,ax=plt.subplots(figsize=(6.4,5.2))
    for model,g in oof.groupby('model'):
        a=g.groupby('study_id').agg(y=('label','first'),p=('probability','mean'))
        a=a.sort_values('p').reset_index(drop=True)
        a['bin']=pd.qcut(a.p,q=min(8,a.p.nunique()),duplicates='drop')
        bins=a.groupby('bin',observed=True).agg(obs=('y','mean'),pred=('p','mean'))
        ax.plot(bins.pred,bins.obs,'o-',label=model)
    ax.plot([0,1],[0,1],ls='--',c='0.6')
    ax.set(xlabel='Average predicted probability',ylabel='Observed BD fraction',
           title='Calibration of mean-patient OOF probability')
    ax.legend(fontsize=8);f.tight_layout()
    f.savefig(OUT_MANUSCRIPT/'calibration_patient_mean_probabilities.png',dpi=220);plt.close(f)
    f,ax=plt.subplots(figsize=(6.2,3.8))
    for model,g in oof.groupby('model'):
        th=g.groupby(['repeat','fold'])['threshold'].first()
        ax.hist(th,alpha=0.4,bins=20,label=model)
    ax.set(xlabel='Fold-selected threshold',ylabel='Count',title='Thresholds selected inside training folds')
    ax.legend(fontsize=8);f.tight_layout()
    f.savefig(OUT_MANUSCRIPT/'thresholds_per_outer_fold.png',dpi=220);plt.close(f)


def write_manifest(data=None):
    packages=['numpy','pandas','scikit-learn','xgboost','scipy','matplotlib','PyMuPDF','openpyxl']
    package_versions={p:(md.version(p) if p in {d.metadata['Name'] for d in md.distributions()} else 'unavailable') for p in packages}
    settings=dict(mode=MODE,seed=SEED,n_splits=N_SPLITS,n_repeats=N_REPEATS,
                  bootstrap_patients=N_BOOT,sensitivity_target=SENSITIVITY_TARGET,
                  curated_whitelist=USE_CURATED_WHITELIST,
                  hyperparameters={name:m.get_params() for name,m in get_models().items()},
                  tfidf_parameters=make_tfidf().get_params(),
                  calibration='Platt sigmoid, internal 3-fold',
                  threshold='inner 3-fold OOF calibrated probabilities',
                  ci_method='patient-cluster bootstrap conditional on CV fits',
                  n_patients=None if data is None else len(data),
                  n_BD=None if data is None else int(data.label.sum()),
                  versions=package_versions,python=sys.version.split()[0])
    (OUT_MANUSCRIPT/'software_and_methods_manifest.json').write_text(json.dumps(settings,indent=2,default=str),encoding='utf-8')
    flow=dict(bd_screened='DOCUMENT FROM RECORDS',bd_excluded='DOCUMENT FROM RECORDS',
              bd_included=None if data is None else int(data.label.sum()),
              nbd_screened='DOCUMENT FROM RECORDS',nbd_excluded='DOCUMENT FROM RECORDS',
              nbd_included=None if data is None else int((data.label==0).sum()),
              note='Distinct sampling pathways; no invented denominators')
    (OUT_MANUSCRIPT/'flowchart_required_counts.json').write_text(json.dumps(flow,indent=2),encoding='utf-8')

# Publishable descriptive Table 1, independent of the ML results.
# This mapping is descriptive only; confirm diag provenance and sex codebook.

def classify_recorded_diagnosis(value):
    s = str(value).strip().upper()
    if s == 'ICH': return 'Intracerebral haemorrhage (ICH)'
    if s in {'IS','ISCHEMIC STROKE'}: return 'Ischaemic stroke'
    if s == 'HIBI': return 'Hypoxic-ischaemic brain injury'
    if s == 'SAH': return 'Subarachnoid haemorrhage (SAH)'
    if s in {'TBI', 'SDH', 'HSD TBI', 'SDH/TBI', 'SAH TBI', 'ICH/SDH TBI'}:
        return 'Traumatic brain injury / subdural'
    return 'Other / mixed diagnoses'


def export_publishable_table1(cohort):
    """Descriptive n/N; unknown sex coding kept verbatim.

    The CT-to-BD interval and the CT-to-recorded-NBD-outcome interval refer
    to different events and must not be compared as identical follow-up.
    """
    rows = []
    groups = {g: cohort.loc[cohort['group_clinical'].eq(g)].copy() for g in ('BD','NBD')}
    for group, data in groups.items():
        n=len(data)
        ages=data['age'].dropna()
        intervals=data['followup_days'].dropna()
        def put(metric,value,denominator=None):
            rows.append({'characteristic':metric,'group':group,'value':value,
                         'denominator':denominator if denominator is not None else n})
        put('Participants',n,n)
        put('Age median (IQR), years', f'{ages.median():.2f} ({ages.quantile(.25):.2f}–{ages.quantile(.75):.2f})',len(ages))
        put('Age missing',int(data['age'].isna().sum()),n)
        for sex in ('1','2'):
            count=int(data['sex'].eq(sex).sum())
            put(f'Sex code {sex} (codebook pending)',f'{count} ({count/n*100:.1f}%)',n)
        cats=data['etiology'].map(classify_recorded_diagnosis)
        for category in ['Intracerebral haemorrhage (ICH)','Ischaemic stroke',
                         'Hypoxic-ischaemic brain injury','Subarachnoid haemorrhage (SAH)',
                         'Traumatic brain injury / subdural','Other / mixed diagnoses']:
            count=int(cats.eq(category).sum())
            put('Recorded diagnosis: '+category,f'{count} ({count/n*100:.1f}%)',n)
        for name,years in [('2011–2012',[2011,2012]),('2013–2021',range(2013,2022)),('2022–2024',[2022,2023,2024])]:
            count=int(data['ct_date'].dt.year.isin(years).sum())
            put('Index CT year: '+name,f'{count} ({count/n*100:.1f}%)',n)
        for cat in ['alta','obito']:
            count=int(data['outcome_type'].eq(cat).sum())
            put('Recorded outcome: '+cat,f'{count} ({count/n*100:.1f}%)',n)
        put('Outcome missing',int(data['outcome_type'].isna().sum()),n)
        if len(intervals):
            put('CT to recorded outcome: median (IQR), days',
                f'{intervals.median():.2f} ({intervals.quantile(.25):.2f}–{intervals.quantile(.75):.2f})',len(intervals))
        put('CT to outcome interval available',len(intervals),n)
    out=pd.DataFrame(rows)
    OUT_MANUSCRIPT.mkdir(exist_ok=True,parents=True)
    out.to_csv(OUT_MANUSCRIPT/'Table1_aggregate_descriptive.csv',index=False)
    return out


def execute(*, fit_models: bool = False, run_shap: bool = False) -> dict:
    """Execute source checks; optionally fit models. Returned frames stay local."""
    if BASE_DIR is None:
        raise RuntimeError('Call configure before execute.')
    cohort = reconcile_source(BASE_DIR, CLINICAL_FILE)
    clinical_descriptives(cohort)
    export_publishable_table1(cohort)
    write_manifest(cohort)
    data = read_pdfs(cohort)
    summarise_manual_negation_accuracy()
    result = {'cohort': cohort, 'data': data}
    if not fit_models:
        print('Cohort and V5 reconciled; predictive models not fitted.')
        return result
    oof, fold_metrics = repeated_cv_predictions(data)
    feature_space_report(data, fold_metrics)
    perf, deltas = patient_cluster_bootstrap(oof, B=N_BOOT)
    calibration = calibration_and_threshold_audit(oof)
    alerts = alert_scenarios(perf)
    export_error_review(oof, cohort)
    figures(oof, perf, calibration)
    if run_shap:
        shap_stability_exploratory(data)
    result.update(oof=oof,fold_metrics=fold_metrics,performance=perf,
                  auc_differences=deltas,calibration=calibration,alerts=alerts)
    print('Outputs:', OUT_MANUSCRIPT)
    return result
