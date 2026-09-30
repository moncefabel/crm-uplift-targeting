"""Analyse de l'expérience randomisée : contrôle SRM, effets moyens, CUPED, puissance."""
import numpy as np

from . import stats as st
from .data import CONTROL, VARIANTS


def analyse(df) -> dict:
    counts = df["segment"].value_counts().reindex([CONTROL] + VARIANTS).values
    out = {"n_total": int(len(df)), "srm_p_value": st.srm_pvalue(counts), "variants": {}}
    ctrl = df[df["segment"] == CONTROL]
    for v in VARIANTS:
        trt = df[df["segment"] == v]
        res = {m: st.diff_in_means(trt[m], ctrl[m]) for m in ["visit", "conversion", "spend"]}
        # CUPED sur la dépense, covariable = dépense de l'année précédente (pré-expérience)
        both = df[df["segment"].isin([v, CONTROL])]
        y_adj, theta = st.cuped(both["spend"], both["history"])
        is_t = (both["segment"] == v).values
        cup = st.diff_in_means(y_adj[is_t], y_adj[~is_t])
        res["spend_cuped"] = cup
        res["cuped_variance_reduction"] = float(1 - (cup["se"] / res["spend"]["se"]) ** 2)
        res["mde_visit_abs"] = st.mde_proportion(ctrl["visit"].mean(), int(min(len(trt), len(ctrl))))
        out["variants"][v] = res
    return out
