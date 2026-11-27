"""Additional analyses ported from the final reviewer notebook (cells 28–34, zero-based).
Run only with authorized study inputs, outside the Git repository.
"""
from pathlib import Path
import argparse
from . import review_analysis as analysis

import numpy as np
import pandas as pd
import json
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, confusion_matrix

DOCUMENTARY_TOKENS = frozenset({
    "ce",
    "tc",
    "estudo",
    "estudos",
    "exame",
    "exames",
    "relatorio",
    "relatorios",
    "requisicao",
    "requisitante",
    "pagina",
    "paginas",
    "assinatura",
    "documento",
    "documentos",
    "informacao",
    "axial",
    "axiais"
})

class V5bTfidfVectorizer(TfidfVectorizer):

    def fit_transform(self, raw_documents, y=None):

        X = super().fit_transform(
            raw_documents, y=y
        )

        names = super().get_feature_names_out()

        keep = np.array([
            not bool(
                DOCUMENTARY_TOKENS.intersection(
                    term.split()
                )
            )
            for term in names
        ], dtype=bool)

        if not keep.any():
            raise ValueError(
                "V5b removeu todas as características."
            )

        self.allowed_feature_idx_ = np.flatnonzero(keep)
        self.n_removed_ = int((~keep).sum())

        return X[:, self.allowed_feature_idx_]

    def transform(self, raw_documents):

        X = super().transform(raw_documents)
        return X[:, self.allowed_feature_idx_]

    def get_feature_names_out(self, input_features=None):

        names = super().get_feature_names_out(
            input_features
        )
        return names[self.allowed_feature_idx_]


def sensitivity_estimator(variant):

    # Mesmos parâmetros da V5 principal
    base_vectorizer = analysis.make_tfidf()

    if variant == "V5":
        vectorizer = base_vectorizer

    elif variant == "V5b":
        vectorizer = V5bTfidfVectorizer(
            **base_vectorizer.get_params(deep=False)
        )

    else:
        raise ValueError(f"Variante desconhecida: {variant}")

    # Mesma LR da análise principal
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", clone(analysis.get_models()["LR"]))
    ])

    # Mesma calibração sigmoide interna
    return CalibratedClassifierCV(
        estimator=pipeline,
        cv=3,
        method="sigmoid",
        ensemble=False
    )




def run_sensitivity(data_extra, out_dir, primary_dir):
    full = (
        data_extra.copy()
        .sort_values("study_id")
        .reset_index(drop=True)
    )

    assert len(full) == 161
    assert full["study_id"].is_unique
    assert full["label"].sum() == 94
    assert (full["label"] == 0).sum() == 67
    assert full["cleaned_preview"].notna().all()

    full["year"] = pd.to_datetime(
        full["ct_date"], errors="coerce"
    ).dt.year

    assert full["year"].notna().all(), (
        "Existem datas de TC em falta."
    )

    # Excluir os BD de 2013–2021.
    # Como não há NBD nesses anos, esta restrição
    # equivale a conservar os anos 2011, 2012, 2022–2024.
    common = (
        full.loc[~full["year"].between(2013, 2021)]
        .copy()
        .reset_index(drop=True)
    )

    assert len(common) == 112
    assert common["label"].sum() == 45
    assert (common["label"] == 0).sum() == 67


    DIR_SENS = out_dir / "sensitivity"
    DIR_SENS.mkdir(parents=True, exist_ok=True)
    DIR_PRIMARY = primary_dir

    SEED = analysis.SEED

    # ============================================================
    # REPEATED 5 x 20 CV — SENSITIVITY ANALYSES
    # LR only | sensitivity target 95%
    # ============================================================

    REPEATS_SENS = 20
    FOLDS_SENS = 5
    BOOTSTRAP_SENS = 2000
    TARGET_SENS = 0.95


    def run_sensitivity_cv(df, variant, name):

        df = df.reset_index(drop=True).copy()

        y = df["label"].to_numpy(dtype=int)
        text = df["cleaned_preview"].astype(str)

        out_dir = DIR_SENS / name
        out_dir.mkdir(parents=True, exist_ok=True)

        cv = RepeatedStratifiedKFold(
            n_splits=FOLDS_SENS,
            n_repeats=REPEATS_SENS,
            random_state=SEED
        )

        predictions = []
        fold_metrics = []

        for k, (tr, te) in enumerate(cv.split(text, y)):

            rep, fold = divmod(k, FOLDS_SENS)

            x_train = text.iloc[tr]
            x_test = text.iloc[te]

            y_train = y[tr]
            y_test = y[te]

            # 1. Previsões calibradas internas,
            # exclusivamente no treino externo
            inner_cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=SEED + 100 * k
            )

            inner_estimator = sensitivity_estimator(variant)

            p_inner = cross_val_predict(
                inner_estimator,
                x_train,
                y_train,
                cv=inner_cv,
                method="predict_proba",
                n_jobs=1
            )[:, 1]

            # 2. Threshold escolhido sem ver o teste
            threshold = analysis.choose_threshold(
                y_train,
                p_inner,
                target=TARGET_SENS
            )

            # 3. Modelo treinado no treino externo
            model = sensitivity_estimator(variant)
            model.fit(x_train, y_train)

            # 4. Previsões no fold de teste
            probabilities = model.predict_proba(x_test)[:, 1]

            predicted = (
                probabilities >= threshold
            ).astype(int)

            fitted_vectorizer = (
                model.calibrated_classifiers_[0]
                .estimator.named_steps["vectorizer"]
            )

            n_features = len(
                fitted_vectorizer.get_feature_names_out()
            )

            n_removed = (
                fitted_vectorizer.n_removed_
                if variant == "V5b"
                else 0
            )

            for pos, prob, pred in zip(
                te, probabilities, predicted
            ):
                predictions.append({
                    "repeat": rep,
                    "fold": fold,
                    "model": "LR",
                    "study_id": df.iloc[pos]["study_id"],
                    "label": int(y[pos]),
                    "probability": float(prob),
                    "threshold": float(threshold),
                    "prediction": int(pred)
                })

            fold_metrics.append({
                "repeat": rep,
                "fold": fold,
                "model": "LR",
                "auc": roc_auc_score(
                    y_test, probabilities
                ),
                "sensitivity": float(
                    predicted[y_test == 1].mean()
                ),
                "specificity": float(
                    (1 - predicted[y_test == 0]).mean()
                ),
                "threshold": float(threshold),
                "n_tfidf_features": n_features,
                "n_removed_features": n_removed
            })

            if (k + 1) % 10 == 0:
                print(
                    f"{name}: {k + 1}/"
                    f"{FOLDS_SENS * REPEATS_SENS} folds"
                )

        oof = pd.DataFrame(predictions)
        folds = pd.DataFrame(fold_metrics)

        assert len(oof) == len(df) * REPEATS_SENS

        assert (
            oof.groupby("study_id").size()
            == REPEATS_SENS
        ).all()

        # Verificação de unicidade OOF
        assert not oof.duplicated(
            ["study_id", "repeat"]
        ).any()

        oof.to_csv(
            out_dir / "oof_predictions_pseudonymous.csv",
            index=False
        )

        folds.to_csv(
            out_dir / "fold_metrics_DESCRIPTIVE.csv",
            index=False
        )

        # Reutilizar funções estatísticas originais,
        # sem escrever sobre a análise principal.
    

        original_out = analysis.OUT_MANUSCRIPT

        try:
            analysis.OUT_MANUSCRIPT = out_dir

            performance, _ = analysis.patient_cluster_bootstrap(
                oof,
                B=BOOTSTRAP_SENS
            )

            calibration = analysis.calibration_and_threshold_audit(oof)
            alerts = analysis.alert_scenarios(performance)

        finally:
            analysis.OUT_MANUSCRIPT = original_out

        # Manifesto da configuração
        manifest = {
            "scenario": name,
            "variant": variant,
            "n_patients": int(len(df)),
            "n_bd": int(y.sum()),
            "n_nbd": int((y == 0).sum()),
            "outer_folds": FOLDS_SENS,
            "outer_repeats": REPEATS_SENS,
            "bootstrap_samples": BOOTSTRAP_SENS,
            "training_sensitivity_target": TARGET_SENS,
            "seed": SEED,
            "classifier": "LR",
            "documentary_exclusions": (
                sorted(DOCUMENTARY_TOKENS)
                if variant == "V5b" else []
            ),
            "analysis_type": "post-hoc sensitivity analysis"
        }

        with open(
            out_dir / "analysis_manifest.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(manifest, f, indent=2)

        return oof, performance, calibration, folds


    # ============================================================
    # EXECUÇÃO
    # ============================================================

    scenarios = {
        "B_common_years_V5": (common, "V5"),
        "C_full_V5b": (full, "V5b"),
        "D_common_years_V5b": (common, "V5b")
    }

    sensitivity_results = {}

    for name, (df_scenario, variant) in scenarios.items():

        print("\n" + "=" * 65)
        print("SCENARIO:", name)
        print("N:", len(df_scenario))
        print("=" * 65)

        sensitivity_results[name] = run_sensitivity_cv(
            df_scenario,
            variant,
            name
        )

    print("\nTRÊS ANÁLISES DE SENSIBILIDADE CONCLUÍDAS.")


    # ============================================================
    # COMPARISON: A vs B vs C vs D
    # ============================================================

    primary = pd.read_csv(
        DIR_PRIMARY / "patient_cluster_bootstrap_performance.csv"
    )

    primary = primary[primary["model"] == "LR"].copy()

    metrics = [
        "auc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "brier"
    ]


    def summarize_performance(df, label, n):

        sub = (
            df[df["model"] == "LR"]
            .set_index("metric")
        )

        row = {
            "scenario": label,
            "n_patients": n
        }

        for metric in metrics:

            row[metric] = float(
                sub.loc[metric, "estimate"]
            )

            row[f"{metric}_CI_lower"] = float(
                sub.loc[metric, "ci_lower"]
            )

            row[f"{metric}_CI_upper"] = float(
                sub.loc[metric, "ci_upper"]
            )

        return row


    rows = [
        summarize_performance(
            primary,
            "A_full_V5_PRIMARY",
            161
        )
    ]

    for name, (df_scenario, variant) in scenarios.items():

        oof, perf, calibration, folds = sensitivity_results[name]

        rows.append(
            summarize_performance(
                perf,
                name,
                len(df_scenario)
            )
        )

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        DIR_SENS / "A_B_C_D_sensitivity_comparison.csv",
        index=False
    )

    print("\nRESULTADOS COMPARATIVOS")
    print(
        comparison[
            [
                "scenario",
                "n_patients",
                "auc",
                "sensitivity",
                "specificity",
                "brier"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nINTERVALOS DE CONFIANÇA DA AUC")
    print(
        comparison[
            [
                "scenario",
                "auc",
                "auc_CI_lower",
                "auc_CI_upper"
            ]
        ].round(4).to_string(index=False)
    )

    print("\nCARACTERÍSTICAS REMOVIDAS PELA V5b")
    for name in [
        "C_full_V5b",
        "D_common_years_V5b"
    ]:
        folds = sensitivity_results[name][3]

        print(
            name,
            "| média de features removidas:",
            round(folds["n_removed_features"].mean(), 1),
            "| média de features finais:",
            round(folds["n_tfidf_features"].mean(), 1)
        )

    print("\nTabela guardada em:")
    print(DIR_SENS / "A_B_C_D_sensitivity_comparison.csv")


def run_temporal(data_extra, out_dir):
    SEED_TEMP = 42
    N_FOLDS_TEMP = 5
    N_REPEATS_TEMP = 20

    DIR_TEMP = out_dir / "temporal"
    DIR_TEMP.mkdir(parents=True, exist_ok=True)

    df_temp = data_extra.copy()

    df_temp["year"] = pd.to_datetime(
        df_temp["ct_date"], errors="coerce"
    ).dt.year

    if df_temp["year"].isna().any():
        raise ValueError("Existem datas de TC em falta.")

    # Períodos definidos antes desta execução
    df_temp = df_temp[
        df_temp["year"].isin([2011, 2012, 2022, 2023, 2024])
    ].copy()

    df_temp["period"] = np.where(
        df_temp["year"] <= 2012,
        "2011-2012",
        "2022-2024"
    )

    df_temp = (
        df_temp.sort_values("study_id")
        .reset_index(drop=True)
    )

    assert len(df_temp) == 112
    assert df_temp["study_id"].is_unique
    assert df_temp["label"].sum() == 45

    # Estratificação conjunta
    df_temp["stratum"] = (
        df_temp["period"]
        + "_"
        + df_temp["label"].astype(str)
    )

    print("DISTRIBUIÇÃO TEMPORAL")
    print(pd.crosstab(df_temp["period"], df_temp["label"]))

    assert df_temp["stratum"].value_counts().min() >= 5

    y = df_temp["label"].to_numpy(dtype=int)
    texts = df_temp["cleaned_preview"].astype(str)

    cv = RepeatedStratifiedKFold(
        n_splits=N_FOLDS_TEMP,
        n_repeats=N_REPEATS_TEMP,
        random_state=SEED_TEMP
    )

    records = []
    fold_records = []

    # A função original utiliza esta variável global
    previous_target = analysis.SENSITIVITY_TARGET

    try:
        analysis.SENSITIVITY_TARGET = 0.95

        for k, (train, test) in enumerate(
            cv.split(texts, df_temp["stratum"])
        ):

            repetition, fold = divmod(k, N_FOLDS_TEMP)

            probability, threshold, n_features = (
                analysis.train_one_outer_fold(
                    texts.iloc[train],
                    y[train],
                    texts.iloc[test],
                    analysis.get_models()["LR"],
                    seed=SEED_TEMP + 100 * k
                )
            )

            predicted = (probability >= threshold).astype(int)

            # Controlo negativo simples: período apenas,
            # estimado exclusivamente no treino externo.
            period_risk = (
                df_temp.iloc[train]
                .groupby("period")["label"].mean()
            )

            era_probability = (
                df_temp.iloc[test]["period"]
                .map(period_risk)
                .to_numpy(float)
            )

            for j, idx in enumerate(test):
                records.append({
                    "repeat": repetition,
                    "fold": fold,
                    "study_id": df_temp.iloc[idx]["study_id"],
                    "period": df_temp.iloc[idx]["period"],
                    "label": int(y[idx]),
                    "probability": float(probability[j]),
                    "prediction": int(predicted[j]),
                    "threshold": float(threshold),
                    "period_only_probability": float(
                        era_probability[j]
                    )
                })

            fold_records.append({
                "repeat": repetition,
                "fold": fold,
                "auc": roc_auc_score(y[test], probability),
                "n_features": n_features,
                "threshold": threshold
            })

            if (k + 1) % 10 == 0:
                print(f"CV concluída: {k + 1}/100 folds")

    finally:
        analysis.SENSITIVITY_TARGET = previous_target

    temporal_oof = pd.DataFrame(records)
    temporal_folds = pd.DataFrame(fold_records)

    # Auditoria OOF
    assert len(temporal_oof) == 112 * 20

    assert (
        temporal_oof.groupby("study_id").size() == 20
    ).all()

    assert not temporal_oof.duplicated(
        ["repeat", "study_id"]
    ).any()

    temporal_oof.to_csv(
        DIR_TEMP / "temporal_stratified_OOF_PRIVATE.csv",
        index=False
    )

    temporal_folds.to_csv(
        DIR_TEMP / "temporal_fold_metrics.csv",
        index=False
    )

    print("\nVALI​DAÇÃO TEMPORAL CONCLUÍDA")
    print("Registos OOF:", len(temporal_oof))
    print("Pasta:", DIR_TEMP)


    # ============================================================
    # TEMPORAL PERFORMANCE + PATIENT-LEVEL BOOTSTRAP
    # ============================================================


    N_BOOT_TEMP = 2000

    ids = df_temp["study_id"].tolist()
    labels = df_temp["label"].to_numpy(int)
    periods = df_temp["period"].to_numpy()

    prob_cube = (
        temporal_oof.pivot(
            index="repeat",
            columns="study_id",
            values="probability"
        )
        .reindex(index=range(20), columns=ids)
        .to_numpy(float)
    )

    pred_cube = (
        temporal_oof.pivot(
            index="repeat",
            columns="study_id",
            values="prediction"
        )
        .reindex(index=range(20), columns=ids)
        .to_numpy(int)
    )

    era_cube = (
        temporal_oof.pivot(
            index="repeat",
            columns="study_id",
            values="period_only_probability"
        )
        .reindex(index=range(20), columns=ids)
        .to_numpy(float)
    )

    assert np.isfinite(prob_cube).all()
    assert np.isfinite(era_cube).all()

    GROUPS_TEMP = [
        "All",
        "2011-2012",
        "2022-2024"
    ]

    def temporal_metrics(idx):

        yy = labels[idx]
        pp = prob_cube[:, idx]
        zz = pred_cube[:, idx]
        ee = era_cube[:, idx]

        positive = (yy == 1)
        negative = (yy == 0)

        return {
            "auc": float(np.mean([
                roc_auc_score(yy, p)
                for p in pp
            ])),
            "period_only_auc": float(np.mean([
                roc_auc_score(yy, p)
                for p in ee
            ])),
            "sensitivity": float(
                zz[:, positive].mean()
            ),
            "specificity": float(
                (1 - zz[:, negative]).mean()
            ),
            "brier": float(
                np.mean((pp - yy[None, :]) ** 2)
            )
        }

    def group_indices(group, indices=None):
        if indices is None:
            indices = np.arange(len(labels))

        if group == "All":
            return indices

        return indices[periods[indices] == group]

    # Estimativas pontuais
    point = {
        g: temporal_metrics(group_indices(g))
        for g in GROUPS_TEMP
    }

    # Amostragem estratificada por período + desfecho
    strata_indices = [
        np.flatnonzero(
            (periods == period) & (labels == label)
        )
        for period in ["2011-2012", "2022-2024"]
        for label in [0, 1]
    ]

    rng = np.random.default_rng(20261108)

    draws = {
        g: {metric: [] for metric in point[g]}
        for g in GROUPS_TEMP
    }

    for b in range(N_BOOT_TEMP):

        sample = np.concatenate([
            rng.choice(
                idx,
                size=len(idx),
                replace=True
            )
            for idx in strata_indices
        ])

        for group in GROUPS_TEMP:

            metrics = temporal_metrics(
                group_indices(group, sample)
            )

            for metric, value in metrics.items():
                draws[group][metric].append(value)

        if (b + 1) % 500 == 0:
            print(f"Bootstrap: {b + 1}/{N_BOOT_TEMP}")

    rows = []

    for group in GROUPS_TEMP:

        n = len(group_indices(group))

        for metric, estimate in point[group].items():

            dist = np.asarray(draws[group][metric])

            rows.append({
                "period": group,
                "n": n,
                "metric": metric,
                "estimate": estimate,
                "ci_lower": np.percentile(dist, 2.5),
                "ci_upper": np.percentile(dist, 97.5)
            })

    temporal_results = pd.DataFrame(rows)

    temporal_results.to_csv(
        DIR_TEMP / "temporal_performance_bootstrap.csv",
        index=False
    )

    # Diferença de discriminação entre períodos
    auc_difference = (
        point["2022-2024"]["auc"]
        - point["2011-2012"]["auc"]
    )

    boot_difference = (
        np.asarray(draws["2022-2024"]["auc"])
        - np.asarray(draws["2011-2012"]["auc"])
    )

    print("\nDESEMPENHO TEMPORAL")
    print(
        temporal_results[
            temporal_results.metric.isin([
                "auc", "period_only_auc",
                "sensitivity", "specificity"
            ])
        ][[
            "period", "metric", "estimate",
            "ci_lower", "ci_upper"
        ]].round(4).to_string(index=False)
    )

    print("\nDIFERENÇA DE AUC: 2022-24 menos 2011-12")
    print("Estimativa:", round(auc_difference, 4))
    print(
        "IC 95%:",
        np.round(
            np.percentile(boot_difference, [2.5, 97.5]),
            4
        )
    )

    pd.DataFrame([{
        "comparison": "2022-2024 minus 2011-2012",
        "auc_difference": auc_difference,
        "ci_lower": np.percentile(boot_difference, 2.5),
        "ci_upper": np.percentile(boot_difference, 97.5)
    }]).to_csv(
        DIR_TEMP / "temporal_auc_difference.csv",
        index=False
    )


    # ============================================================
    # CROSS-PERIOD TRANSPORTABILITY
    # Train old -> Test recent
    # Train recent -> Test old
    # ============================================================


    transfer_results = []
    transfer_predictions = []

    previous_target = analysis.SENSITIVITY_TARGET

    try:
        analysis.SENSITIVITY_TARGET = 0.95

        comparisons = [
            ("2011-2012", "2022-2024"),
            ("2022-2024", "2011-2012")
        ]

        for k, (train_period, test_period) in enumerate(
            comparisons
        ):

            train = df_temp[
                df_temp.period == train_period
            ].copy()

            test = df_temp[
                df_temp.period == test_period
            ].copy()

            y_train = train.label.to_numpy(int)
            y_test = test.label.to_numpy(int)

            probabilities, threshold, n_features = (
                analysis.train_one_outer_fold(
                    train.cleaned_preview.astype(str),
                    y_train,
                    test.cleaned_preview.astype(str),
                    analysis.get_models()["LR"],
                    seed=SEED_TEMP + 10000 + k
                )
            )

            predictions = (
                probabilities >= threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).ravel()

            auc = roc_auc_score(y_test, probabilities)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # IC bootstrap sobre o período de teste
            rng = np.random.default_rng(42 + k)

            pos = np.where(y_test == 1)[0]
            neg = np.where(y_test == 0)[0]

            auc_draws = []

            for _ in range(2000):

                idx = np.concatenate([
                    rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)
                ])

                auc_draws.append(
                    roc_auc_score(
                        y_test[idx], probabilities[idx]
                    )
                )

            transfer_results.append({
                "training_period": train_period,
                "testing_period": test_period,
                "n_train": len(train),
                "n_test": len(test),
                "auc": auc,
                "ci_lower": np.percentile(auc_draws, 2.5),
                "ci_upper": np.percentile(auc_draws, 97.5),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "threshold": threshold,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            })

            for sid, yy, pp, pred in zip(
                test.study_id,
                y_test,
                probabilities,
                predictions
            ):
                transfer_predictions.append({
                    "training_period": train_period,
                    "testing_period": test_period,
                    "study_id": sid,
                    "label": int(yy),
                    "probability": float(pp),
                    "prediction": int(pred)
                })

    finally:
        analysis.SENSITIVITY_TARGET = previous_target

    transfer_df = pd.DataFrame(transfer_results)

    transfer_df.to_csv(
        DIR_TEMP / "cross_period_transfer_summary.csv",
        index=False
    )

    pd.DataFrame(transfer_predictions).to_csv(
        DIR_TEMP / "cross_period_transfer_OOF_PRIVATE.csv",
        index=False
    )

    print("\nCROSS-PERIOD TRANSFER")
    print(
        transfer_df[
            [
                "training_period",
                "testing_period",
                "n_train",
                "n_test",
                "auc",
                "ci_lower",
                "ci_upper",
                "sensitivity",
                "specificity"
            ]
        ].round(4).to_string(index=False)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Post-hoc study sensitivity and temporal analyses")
    for name in ("base-dir", "clinical-file", "v5-csv", "out-dir", "primary-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    analysis.configure(base_dir=args.base_dir, clinical_file=args.clinical_file,
                       v5_csv=args.v5_csv, out_dir=args.out_dir / "reconciliation",
                       mode="final", target=0.95)
    data = analysis.read_pdfs(analysis.reconcile_source(analysis.BASE_DIR, analysis.CLINICAL_FILE))
    run_sensitivity(data, args.out_dir, args.primary_dir)
    run_temporal(data, args.out_dir)

if __name__ == "__main__":
    main()
