"""Chargement et préparation du jeu de données Hillstrom (campagne e-mail randomisée, 64 000 clients)."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hillstrom.csv"
CONTROL = "No E-Mail"
VARIANTS = ["Mens E-Mail", "Womens E-Mail"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"recency", "history", "mens", "womens", "zip_code", "newbie",
                "channel", "segment", "visit", "conversion", "spend"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables pré-traitement uniquement (aucune fuite d'information post-campagne)."""
    X = df[["recency", "history", "mens", "womens", "newbie"]].copy()
    X = X.join(pd.get_dummies(df[["zip_code", "channel"]], drop_first=True, dtype=int))
    return X


def binary_experiment(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Sous-échantillon variante contre témoin, avec indicateur de traitement."""
    sub = df[df["segment"].isin([variant, CONTROL])].copy()
    sub["treatment"] = (sub["segment"] == variant).astype(int)
    return sub.reset_index(drop=True)
