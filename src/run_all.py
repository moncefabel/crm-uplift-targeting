"""Point d'entrée : python -m src.run_all"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklift.metrics import qini_curve

from . import ab_test, uplift
from .data import VARIANTS, binary_experiment, load_raw

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "reports" / "figures", ROOT / "results"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    # 1. Analyse de l'A/B test
    ab = ab_test.analyse(df)
    (RES / "ab_test.json").write_text(json.dumps(ab, indent=2))

    # 2. Uplift modeling, une expérience binaire par variante
    all_rows = {}
    for v in VARIANTS:
        exp = binary_experiment(df, v)
        rows = uplift.benchmark(exp, target="visit")
        base = uplift.random_baseline(exp, target="visit")
        all_rows[v] = {"models": rows, "random": base}
        pd.DataFrame(rows).round(4).to_csv(RES / f"uplift_{v.split()[0].lower()}.csv", index=False)
    (RES / "uplift.json").write_text(json.dumps(all_rows, indent=2))

    # 3. Figures sur la variante à effet le plus hétérogène (e-mail Femmes)
    exp = binary_experiment(df, "Womens E-Mail")
    rows = pd.DataFrame(all_rows["Womens E-Mail"]["models"])
    best = rows.sort_values("qini_auc", ascending=False).iloc[0]
    X_tr, X_te, y_tr, y_te, t_tr, t_te = uplift.split(exp, "visit", seed=0)
    model = uplift.APPROACHES[best["approach"]](uplift.BASE_LEARNERS[best["base_learner"]], 0)
    u = model.fit(X_tr, y_tr, t_tr).predict(X_te)

    xs, ys = qini_curve(y_te, u, t_te)
    plt.figure(figsize=(7, 4))
    plt.plot(xs / len(y_te) * 100, ys, label=f"{best['approach']} ({best['base_learner']})")
    plt.plot([0, 100], [0, ys[-1]], ls="--", c="grey", label="Ciblage aléatoire")
    plt.xlabel("Part de clients ciblés (%)"); plt.ylabel("Visites incrémentales cumulées")
    plt.title("Courbe de Qini, e-mail Femmes (jeu de test)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "qini_curve.png", dpi=150); plt.close("all")

    dec = pd.qcut(pd.Series(u).rank(method="first", ascending=False), 10, labels=range(1, 11))
    up = [y_te[(dec == d) & (t_te == 1)].mean() - y_te[(dec == d) & (t_te == 0)].mean() for d in range(1, 11)]
    plt.figure(figsize=(7, 4))
    plt.bar(range(1, 11), np.array(up) * 100)
    plt.axhline((y_te[t_te == 1].mean() - y_te[t_te == 0].mean()) * 100, ls="--", c="k", label="Effet moyen")
    plt.xlabel("Décile d'uplift prédit (1 = plus fort)"); plt.ylabel("Uplift observé (points de %)")
    plt.title("Uplift observé par décile, e-mail Femmes, taux de visite"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "uplift_by_decile.png", dpi=150); plt.close("all")

    plt.figure(figsize=(7, 4))
    for _, r in rows.iterrows():
        plt.scatter(r["energy_wh"], r["qini_auc"], s=60)
        plt.annotate(f"{r['approach']}\n{r['base_learner']}", (r["energy_wh"], r["qini_auc"]), fontsize=7)
    plt.xlabel("Énergie estimée, 5 entraînements (Wh, CodeCarbon)"); plt.ylabel("Qini AUC")
    plt.title("Performance contre sobriété, e-mail Femmes"); plt.tight_layout()
    plt.savefig(FIG / "qini_vs_energy.png", dpi=150); plt.close("all")

    print(json.dumps({"ab": {v: {k: ab["variants"][v][k] for k in ["cuped_variance_reduction", "mde_visit_abs"]}
                             for v in VARIANTS}, "srm": ab["srm_p_value"]}, indent=2))
    print(rows[["approach", "base_learner", "qini_auc", "qini_auc_std", "uplift_at_k", "capture_at_k",
                "energy_wh", "seconds"]].round(4).to_string())


if __name__ == "__main__":
    main()
