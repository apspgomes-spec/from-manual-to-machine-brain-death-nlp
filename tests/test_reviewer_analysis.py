"""Synthetic-only checks: no original clinical files required."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bd_ct_nlp import review_analysis as ra
from bd_ct_nlp.metadata_controls import metadata_features, control_estimator
from bd_ct_nlp.visualization import descriptive_frequencies


def test_threshold_is_max_specificity_under_sensitivity_constraint():
    y=np.array([1,1,1,0,0,0])
    p=np.array([.9,.7,.2,.6,.4,.1])
    t=ra.choose_threshold(y,p,target=1)
    pred=p>=t
    assert pred[y==1].all()
    assert t==.2
    assert (~pred[y==0]).sum()==1
    t95=ra.choose_threshold(y,p,target=.95)
    assert np.isclose(t95,t)


def test_tfidf_factory_no_fitting_or_curated_list():
    vectorizer=ra.make_tfidf()
    assert vectorizer.ngram_range==(1,3)
    assert vectorizer.min_df==2
    assert vectorizer.max_features==5000
    assert not hasattr(vectorizer,'vocabulary_')


def test_display_filter_excludes_admin_without_creating_new_ngrams():
    docs=pd.Series(['sem desvio da linha media', 'efeito de massa',
                    'efeito massa', 'estudo cerca nao'])
    one=descriptive_frequencies(docs,(1,1))
    assert 'cerca' not in one and 'estudo' not in one and 'nao' not in one
    pair=descriptive_frequencies(docs,(2,2))
    assert pair['efeito massa']==1
    assert 'efeito de' not in pair and 'de massa' not in pair
    assert 'desvio linha' not in pair


def test_metadata_features_and_model():
    x=pd.DataFrame(dict(ct_date=pd.to_datetime(['2011-01-01','2024-01-01']),
                        template=['CER','STD'],clinical_words=[45,88],
                        nlp_text=['efeito massa','linha media']))
    f=metadata_features(x)
    assert f['report_year'].tolist()==['2011','2024']
    assert np.isfinite(f.log_report_length).all()
    assert control_estimator('All_metadata').named_steps['features'] is not None


def test_v5_join_discards_leftover_indicator(tmp_path):
    ra.configure(base_dir=tmp_path,clinical_file=tmp_path/'unused.xlsx',
                 v5_csv=tmp_path/'v5.csv',out_dir=tmp_path/'private',mode='test')
    base=pd.DataFrame([dict(pdf_name=f'{i}.pdf',_merge='both',
                             group_pdf='BD' if i==1 else 'NBD',
                             label=1 if i==1 else 0,study_id=f'P{i:04d}')
                       for i in range(1,162)])
    base['group_pdf']='BD';base['label']=1
    base.loc[94:,'group_pdf']='NBD';base.loc[94:,'label']=0
    raw=pd.DataFrame([dict(filename=f'{i}.pdf',group='BD' if i<=94 else 'NBD',
                           label=1 if i<=94 else 0,nlp_text='efeito massa',
                           template='STANDARD_REPORT_SECTION') for i in range(1,162)])
    raw.to_csv(tmp_path/'v5.csv',index=False)
    result=ra.read_pdfs(base)
    assert len(result)==161
    assert '_merge' not in result.columns
