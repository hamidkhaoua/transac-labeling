"""
API de classification de transactions.

Le modele est charge depuis model.joblib (genere par le notebook). S'il est
absent ou incompatible, il est reentraine puis sauvegarde automatiquement.

Lancer :  uvicorn app:app --port 8000
Tester :  http://127.0.0.1:8000/docs
"""
import os
import re

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE, "model.joblib")


def clean_text(x):
    x = str(x).lower()
    x = re.sub(r"\b(paypal|sq|sumup|ztl|zettle|stripe)\s*\*", " ", x)  # processeurs
    x = re.sub(r"[^a-z\s]", " ", x)                                    # ponctuation + chiffres
    return re.sub(r"\s+", " ", x).strip()


def make_features(d):
    d = d.copy()
    d["clean_title"] = d["title"].fillna("").map(clean_text)
    amt_raw = pd.to_numeric(d["amount_value"], errors="coerce")
    d["amount_isnan"] = amt_raw.isna().astype(int)
    amt = amt_raw.fillna(0)
    d["is_credit"] = (amt > 0).astype(int)
    status = d["status"] if "status" in d else pd.Series(["completed"] * len(d))
    d["is_refunded"] = (status.astype(str) == "refunded").astype(int)
    d["log_amount"] = np.log1p(amt.abs())
    return d


def train():
    """Entraine le modele a partir des donnees (fallback si model.joblib absent)."""
    df = make_features(pd.read_csv(DATA))
    y = df["minor_category"]
    mapping = (df.dropna(subset=["major_category"])
                 .groupby("minor_category")["major_category"]
                 .agg(lambda s: s.mode().iat[0]).to_dict())
    mapping["refund"] = None

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, analyzer="char_wb", ngram_range=(3, 5))),
        ("svd", TruncatedSVD(n_components=200, random_state=42)),
    ])
    pre = ColumnTransformer([
        ("text", text_pipe, "clean_title"),
        ("num", "passthrough", ["is_credit", "is_refunded", "log_amount", "amount_isnan"]),
    ])
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=16,
                                      class_weight="balanced_subsample",
                                      random_state=42, n_jobs=-1)),
    ]).fit(df, y)
    return model, mapping


# Charge le modele sauvegarde ; sinon (absent ou version incompatible) reentraine.
try:
    bundle = joblib.load(MODEL_PATH)
    MODEL, MINOR_TO_MAJOR = bundle["model"], bundle["minor_to_major"]
    print("Modele charge depuis model.joblib")
except Exception:
    print("model.joblib introuvable ou incompatible -> entrainement...")
    MODEL, MINOR_TO_MAJOR = train()
    joblib.dump({"model": MODEL, "minor_to_major": MINOR_TO_MAJOR}, MODEL_PATH)
    print("Modele entraine et sauvegarde dans model.joblib")

app = FastAPI(title="Transaction labeling")


class Transaction(BaseModel):
    title: str
    amount_value: float = 0.0
    status: str = "completed"


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(txn: Transaction):
    row = make_features(pd.DataFrame([txn.model_dump()]))
    minor = MODEL.predict(row)[0]
    proba = float(MODEL.predict_proba(row)[0].max())
    return {
        "minor_category": minor,
        "major_category": MINOR_TO_MAJOR.get(minor),
        "confidence": round(proba, 3),
    }
