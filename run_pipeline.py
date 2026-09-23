"""ML pipeline: clean -> feature engineering -> KMeans -> save results for the dashboard."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAW = Path("data/raw/clients.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

if not RAW.exists():
    raise SystemExit("data/raw/clients.csv not found. Run:  python generate_data.py")

# ---------------------------------------------------------------- 1. Load & clean
df = pd.read_csv(RAW)
print(f"Loaded {len(df)} rows")
df = df.drop_duplicates(subset="client_id").copy()
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
today = pd.Timestamp.today().normalize()
df["age"] = ((today - df["date_of_birth"]).dt.days / 365.25).round().clip(18, 90)
df["age"] = df["age"].fillna(df["age"].median())
df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())
for c in ["client_type", "gender", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel"]:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()
print(f"After cleaning: {len(df)} unique clients")

# ---------------------------------------------------------------- 2. Features
df["is_investor"] = (df["acquisition_purpose"].str.lower() == "investment").astype(int)
df["has_loan"] = (df["loan_applied"].str.lower().isin(["yes", "y", "true", "1"])).astype(int)
df["is_corporate"] = (df["client_type"].str.lower() == "corporate").astype(int)
df["age_group"] = pd.cut(df["age"], [0, 30, 40, 50, 60, 120],
                         labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)

# Behavioural features drive the clusters. Region is kept OUT of clustering so the
# geographic dashboard can show where each segment lives (not just re-draw the regions).
num = df[["age", "satisfaction_score", "is_investor", "has_loan", "is_corporate"]].astype(float)
num_s = StandardScaler().fit_transform(num)
num_w = num_s * np.array([1.0, 0.8, 1.6, 1.4, 1.4])          # emphasise investment/financing/type
channel = pd.get_dummies(df["referral_channel"], prefix="ch").astype(float) * 0.5
X = pd.concat([pd.DataFrame(num_w, columns=num.columns, index=df.index), channel], axis=1)
Xs = X.values

# ---------------------------------------------------------------- 3. Choose k
scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    scores[k] = float(silhouette_score(Xs, km.labels_, sample_size=2000, random_state=42))
    print(f"k={k}  silhouette={scores[k]:.3f}  inertia={km.inertia_:.0f}")
# prefer 4-6 segments for business usefulness; pick the best silhouette within that range
best_k = max(range(4, 7), key=lambda k: scores[k])
print(f"Chosen k = {best_k}")

km = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(Xs)
df["cluster"] = km.labels_

# ---------------------------------------------------------------- 4. Name segments
prof = df.groupby("cluster").agg(inv=("is_investor", "mean"), loan=("has_loan", "mean"),
                                 corp=("is_corporate", "mean"), age=("age", "mean"),
                                 sat=("satisfaction_score", "mean"))
names, used = {}, set()
def unique(name, r):
    if name in used:                       # disambiguate with a distinguishing trait
        name = f"{name} ({'Older' if r.age >= df['age'].mean() else 'Younger'})"
    i, base = 2, name
    while name in used:
        name, i = f"{base} {i}", i + 1
    used.add(name)
    return name
for c, r in prof.iterrows():
    if r.corp > .5:
        n = "Corporate Investors" if r.inv > .5 else "Corporate Buyers"
    elif r.inv > .5:
        n = "Leveraged Investors" if r.loan > .5 else "Cash Investors"
    else:
        n = "Mortgage Home Buyers" if r.loan > .5 else "Cash Home Buyers"
    names[c] = unique(n, r)
df["segment"] = df["cluster"].map(names)

# ---------------------------------------------------------------- 5. 2-D projection & save
pca = PCA(n_components=2, random_state=42).fit(Xs)
coords = pca.transform(Xs)
df["pca_1"], df["pca_2"] = coords[:, 0], coords[:, 1]

df.drop(columns=["date_of_birth"]).to_csv(OUT / "clients_segmented.csv", index=False)
json.dump({"chosen_k": int(best_k), "silhouette_by_k": scores,
           "explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
           "segment_names": {int(k): v for k, v in names.items()}},
          open(OUT / "model_metrics.json", "w"), indent=2)
joblib.dump({"kmeans": km, "features": list(X.columns)}, "models/kmeans_model.joblib")
print("\nSegment sizes:\n", df["segment"].value_counts().to_string())
print("\nDone. Now run:  streamlit run app.py")
