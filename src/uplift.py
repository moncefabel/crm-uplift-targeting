"""Modèles d'uplift (S-learner, T-learner, transformation de classe) évalués par Qini / AUUC,
avec mesure de l'énergie consommée par CodeCarbon pour arbitrer performance et sobriété."""
import logging
import time

import numpy as np
from codecarbon import OfflineEmissionsTracker
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score
from sklift.models import ClassTransformation, SoloModel, TwoModels

from .data import features

logging.getLogger("codecarbon").setLevel(logging.ERROR)

BASE_LEARNERS = {
    "logistic": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed),
}
APPROACHES = {
    "S-learner": lambda base, seed: SoloModel(estimator=base(seed)),
    "T-learner": lambda base, seed: TwoModels(estimator_trmnt=base(seed), estimator_ctrl=base(seed),
                                              method="vanilla"),
    "Class transformation": lambda base, seed: ClassTransformation(estimator=base(seed)),
}


def split(df, target: str, seed: int, test_size: float = 0.3):
    X, y, t = features(df), df[target].values, df["treatment"].values
    strat = t * 2 + y  # stratification jointe traitement x résultat
    return train_test_split(X, y, t, test_size=test_size, random_state=seed, stratify=strat)


def incremental_capture(y, uplift, t, k: float) -> float:
    """Part des gains incrémentaux captés en ne ciblant que les k % de clients au plus fort uplift prédit."""
    order = np.argsort(-uplift)
    top = order[: int(len(order) * k)]
    def incr(idx):
        yt, yc = y[idx][t[idx] == 1], y[idx][t[idx] == 0]
        return (yt.mean() - yc.mean()) * len(idx)
    return float(incr(top) / incr(np.arange(len(y))))


def benchmark(df, target: str = "visit", seeds=(0, 1, 2, 3, 4), k: float = 0.3) -> list:
    rows = []
    for base_name, base in BASE_LEARNERS.items():
        for app_name, make_model in APPROACHES.items():
            scores = {"qini_auc": [], "uplift_auc": [], "uplift_at_k": [], "capture_at_k": []}
            tracker = OfflineEmissionsTracker(country_iso_code="FRA", save_to_file=False,
                                              log_level="error")
            tracker.start()
            t0 = time.perf_counter()
            for seed in seeds:
                X_tr, X_te, y_tr, y_te, t_tr, t_te = split(df, target, seed)
                model = make_model(base, seed).fit(X_tr, y_tr, t_tr)
                u = model.predict(X_te)
                scores["qini_auc"].append(qini_auc_score(y_te, u, t_te))
                scores["uplift_auc"].append(uplift_auc_score(y_te, u, t_te))
                scores["uplift_at_k"].append(uplift_at_k(y_te, u, t_te, strategy="overall", k=k))
                scores["capture_at_k"].append(incremental_capture(y_te, u, t_te, k))
            elapsed = time.perf_counter() - t0
            kg = tracker.stop() or 0.0
            energy_kwh = float(tracker.final_emissions_data.energy_consumed)
            row = {"approach": app_name, "base_learner": base_name, "seconds": elapsed,
                   "energy_wh": energy_kwh * 1000, "co2_g": kg * 1000}
            for m, vals in scores.items():
                row[m] = float(np.mean(vals))
                row[m + "_std"] = float(np.std(vals))
            rows.append(row)
    return rows


def random_baseline(df, target: str = "visit", seeds=(0, 1, 2, 3, 4), k: float = 0.3) -> dict:
    vals = []
    for seed in seeds:
        _, _, _, y_te, _, t_te = split(df, target, seed)
        u = np.random.default_rng(seed).random(len(y_te))
        vals.append(uplift_at_k(y_te, u, t_te, strategy="overall", k=k))
    return {"uplift_at_k": float(np.mean(vals))}
