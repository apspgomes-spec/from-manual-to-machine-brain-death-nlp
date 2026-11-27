"""Descriptive only: report-length panels and filtered n-gram word clouds.

Visualization filters do not change the NLP representation used by the models.
Figures require manual review for residual identifiable or administrative terms.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

STOPWORDS = set(('''a o as os de da do das dos e em um uma uns umas para por com que se
 ao aos na no nas nos sua seu suas seus sendo mais menos este esta estes estas
 esse essa esses essas aquele aquela pelo pela pelos pelas entre sobre sob ate
 como tambem ainda foi foram ser sao ha havia tem tendo ou nem ja
 sem nao nunca jamais nenhum nenhuma nenhuns nenhumas ausencia inexistencia
 informacao clinico clinica estudo estudos exame exames relatorio relatorios
 pedido requisicao requisitante observacoes observacao tecnica tecnico realizado
 realizada realizados realizadas efetuado efetuada hospital hospitalar servico
 servicos medico medica utente identificacao processo numero pagina paginas folha
 assinatura assinado data hora documento documentos cerca aproximadamente
 sensivelmente''').split())

ADMIN_PATTERNS = tuple(re.compile(p,re.I) for p in (
    r'\binformacao\s+clinica\b',r'\bpara\s+estudo\b',
    r'\bexame\s+realizado\b',r'\brelatorio\s+radiologico\b',
    r'\bnumero\s+do?\s+processo\b',r'\bmedico\s+requisitante\b',
    r'\bservico\s+requisitante\b',
))


def normalize_for_display(text: str)->str:
    text=unicodedata.normalize('NFKD',str(text).lower())
    return ''.join(c for c in text if not unicodedata.combining(c))


def descriptive_frequencies(texts, ngram_range=(1,1), min_df=1):
    """Count native n-grams, then filter phrases (do not create artificial adjacency)."""
    from sklearn.feature_extraction.text import CountVectorizer
    docs=[]
    for original in texts:
        t=normalize_for_display('' if pd.isna(original) else original)
        for pat in ADMIN_PATTERNS:
            t=pat.sub(' ',t)
        docs.append(t)
    if not docs or not any(re.search('[a-z]{3}',t) for t in docs):
        return Counter()
    cv=CountVectorizer(ngram_range=ngram_range,min_df=min_df,
                       token_pattern=r'(?u)\b[a-z]{2,}\b')
    try: X=cv.fit_transform(docs)
    except ValueError: return Counter()
    totals=np.asarray(X.sum(axis=0)).ravel()
    terms=cv.get_feature_names_out()
    return Counter({s:int(n) for s,n in zip(terms,totals)
                    if all(len(w)>=3 and w not in STOPWORDS for w in s.split())})


def render_wordclouds(v5_csv:Path,output_dir:Path,expected_bd:int=94,
                      expected_nbd:int=67)->dict:
    """Generate 6-panel Figure 1 and supplementary phrase clouds. Data remain local."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    v5=pd.read_csv(v5_csv,encoding='utf-8-sig',usecols=['filename','group','clinical_text','nlp_text'])
    v5['group']=v5['group'].astype(str).str.upper().str.strip()
    counts=v5['group'].value_counts().to_dict()
    if counts!={'BD':expected_bd,'NBD':expected_nbd}:
        raise ValueError(f'Unexpected BD/NBD counts: {counts}')
    if v5['filename'].duplicated().any():
        raise ValueError('Duplicate report filenames')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    def image(freq):
        if not freq: raise ValueError('No terms after filtering')
        return WordCloud(width=1200,height=750,background_color='white',
                         max_words=80,collocations=False,random_state=42).generate_from_frequencies(freq)
    f,axes=plt.subplots(3,2,figsize=(14,15))
    f2,ax2=plt.subplots(1,2,figsize=(16,7))
    audit=[]
    for col,group in enumerate(('BD','NBD')):
        sub=v5.loc[v5.group.eq(group)]
        before=sub['clinical_text'].fillna('').astype(str).str.split().str.len()
        after=sub['nlp_text'].fillna('').astype(str).str.split().str.len()
        ax=axes[0,col]
        ax.hist(before,bins=20,alpha=.55,label='Extracted clinical section')
        ax.hist(after,bins=20,alpha=.55,label='NLP V5')
        ax.set(xlabel='Number of words',ylabel='Reports',title=f'{group}: report length')
        ax.legend(frameon=False)
        for row,(label,colname) in enumerate((('Clinical section','clinical_text'),('NLP V5','nlp_text')),1):
            freq=descriptive_frequencies(sub[colname],(1,1))
            axes[row,col].imshow(image(freq));axes[row,col].axis('off')
            axes[row,col].set_title(f'{group}: {label}')
            audit.extend(dict(group=group,stage=label,term=term,n=1,count=count)
                         for term,count in freq.items())
        phrase=descriptive_frequencies(sub['nlp_text'],(2,3),min_df=2)
        ax2[col].imshow(image(phrase));ax2[col].axis('off')
        ax2[col].set_title(f'{group}: NLP V5 bigrams and trigrams')
        audit.extend(dict(group=group,stage='NLP V5',term=term,n=len(term.split()),count=count)
                     for term,count in phrase.items())
    for ax,letter in zip(axes.flat,'ABCDEF'):
        ax.text(-.03,1.03,letter,transform=ax.transAxes,fontweight='bold',fontsize=14)
    f.tight_layout(); f2.tight_layout()
    for name,fig in [('Figure1_wordclouds_v5',f),('Supplementary_ngrams_v5',f2)]:
        fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(audit).to_csv(out/'visualization_frequencies_LOCAL.csv',index=False)
    pd.DataFrame({'word':sorted(STOPWORDS)}).to_csv(out/'visualization_stopwords.csv',index=False)
    return {'n':len(v5),'group_counts':counts,'figures':[str(out/'Figure1_wordclouds_v5.png'),str(out/'Supplementary_ngrams_v5.png')]}
