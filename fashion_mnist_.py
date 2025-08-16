# -*- coding: utf-8 -*-
"""
Fashion MNIST – Local (PyCharm) version
Phases 1-4: EDA, PCA/SVD, Clustering, Analysis
"""

# ============== Setup & Imports ==============
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Dataset
from tensorflow.keras.datasets import fashion_mnist

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Images
from PIL import Image
from io import BytesIO
import base64  # kept (used by array_to_base64 if you later need it)

# Sklearn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

# Class label names (Fashion MNIST)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ============== Phase 1: Data Understanding & Preprocessing ==============
# Load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Train images:", X_train.shape, "Train labels:", y_train.shape)
print("Test  images:", X_test.shape,  "Test  labels:", y_test.shape)
print("Dtype:", X_train.dtype)

# Sanity checks
assert X_train.ndim == 3 and X_train.shape[1:] == (28, 28)
assert X_test.ndim  == 3 and X_test.shape[1:]  == (28, 28)
assert set(np.unique(y_train)).issubset(set(range(10)))

# Metadata
meta = {
    "n_train": len(X_train),
    "n_test": len(X_test),
    "img_height": 28,
    "img_width": 28,
    "n_classes": 10,
}
print("Quick metadata:")
for k, v in meta.items():
    print(f"  - {k}: {v}")

# ---------- Examples per class (single mosaic) ----------
def array_to_base64(img_array: np.ndarray) -> str:
    img = Image.fromarray(np.uint8(img_array), mode="L").resize((84, 84), Image.NEAREST)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def show_examples_mosaic(X, y, class_names, n_per_class=3, pad=6, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    rows, cols = len(class_names), n_per_class
    H, W = X.shape[1], X.shape[2]

    mosaic_h = rows * H + (rows - 1) * pad
    mosaic_w = cols * W + (cols - 1) * pad
    mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)

    for r in range(rows):
        idxs = np.where(y == r)[0]
        pick = rng.choice(idxs, size=min(n_per_class, len(idxs)), replace=False)
        for c, idx in enumerate(pick):
            top = r * (H + pad)
            left = c * (W + pad)
            mosaic[top:top+H, left:left+W] = X[idx]

    fig = px.imshow(mosaic, color_continuous_scale="gray", zmin=0, zmax=255,
                    title="Examples per class (training set) — mosaic")
    fig.update_coloraxes(showscale=False)

    annotations = []
    for r in range(rows):
        y_center = r * (H + pad) + H / 2
        annotations.append(dict(
            x=-10, y=y_center, xref="x", yref="y", text=class_names[r], showarrow=False,
            xanchor="right", yanchor="middle", font=dict(size=12, color="purple")
        ))
    fig.update_layout(annotations=annotations, margin=dict(l=100, r=10, t=60, b=10),
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    fig.show()

show_examples_mosaic(X_train, y_train, CLASS_NAMES, n_per_class=3)

# ---------- Class distribution ----------
train_counts = pd.Series(y_train).value_counts().sort_index()
counts_df = pd.DataFrame({
    "label": train_counts.index,
    "count": train_counts.values,
    "class": [CLASS_NAMES[i] for i in train_counts.index]
})
fig = px.bar(counts_df, x="class", y="count", text="count",
             title="Class distribution – Training set",
             color="class", color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_traces(textposition="outside")
fig.update_layout(xaxis_title="", yaxis_title="Count", xaxis_tickangle=-30)
fig.show()

# ---------- Pixel intensity histogram (raw) ----------
max_images_for_hist = 4000
sample_idxs = np.random.choice(len(X_train), size=min(max_images_for_hist, len(X_train)), replace=False)
sample_pixels_raw = X_train[sample_idxs].reshape(-1)
fig = px.histogram(sample_pixels_raw, nbins=30,
                   title="Raw pixel intensity histogram (sampled)",
                   color_discrete_sequence=["#FF7F50"])
fig.update_layout(xaxis_title="Pixel value (0–255)", yaxis_title="Frequency")
fig.show()

# ---------- Mean & std images (raw) ----------
mean_img_raw = X_train.mean(axis=0)
std_img_raw  = X_train.std(axis=0)
px.imshow(mean_img_raw, color_continuous_scale="Viridis", title="Mean image (raw)").update_coloraxes(showscale=True).show()
px.imshow(std_img_raw,  color_continuous_scale="Plasma", title="Std-dev image (raw)").update_coloraxes(showscale=True).show()

# ---------- Normalize ----------
X_train_norm = X_train.astype(np.float32) / 255.0
X_test_norm  = X_test.astype(np.float32) / 255.0
sample_pixels_norm = X_train_norm[sample_idxs].reshape(-1)
px.histogram(sample_pixels_norm, nbins=30,
             title="Pixel intensity histogram after normalization (sampled)",
             color_discrete_sequence=["#6A5ACD"]).show()

mean_img_norm = X_train_norm.mean(axis=0)
std_img_norm  = X_train_norm.std(axis=0)
px.imshow(mean_img_norm, color_continuous_scale="Cividis", title="Mean image (normalized)").show()
px.imshow(std_img_norm, color_continuous_scale="Inferno", title="Std-dev image (normalized)").show()

# ---------- Flatten ----------
X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
X_test_flat  = X_test_norm.reshape(len(X_test_norm), -1)
print("Flattened shapes:", X_train_flat.shape, X_test_flat.shape)

# ---------- Random preview ----------
idx = np.random.randint(0, len(X_train_norm))
px.imshow(X_train_norm[idx], color_continuous_scale="Magma",
          title=f"Random normalized sample — label={y_train[idx]} ({CLASS_NAMES[y_train[idx]]})").show()

print("\nPhase 1 complete ✅ — Data loaded, explored with Plotly, normalized, and flattened.")

# ============== Phase 2: PCA & SVD Analysis ==============

# 1) PCA – Fit and Explained Variance
pca = PCA()
pca.fit(X_train_flat)

explained_var_ratio = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var_ratio)

fig = go.Figure()
fig.add_trace(go.Scatter(y=explained_var_ratio, mode="lines+markers", name="Explained Variance"))
fig.add_trace(go.Scatter(y=cum_explained_var,   mode="lines+markers", name="Cumulative Explained Variance"))
fig.update_layout(title="PCA Explained Variance", xaxis_title="Number of Components", yaxis_title="Variance Ratio")
fig.show()

# Precompute full PCA transform for later phases (avoid re-fit)
X_train_pca = pca.transform(X_train_flat)  # <— this was missing before

# 2) PCA reconstructions at k = 10, 50, 100 (+ MSE)
def pca_reconstruct(pca_model: PCA, X_flat: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct X_flat using the first k principal components (manual back-projection)."""
    Z = np.dot(X_flat - pca_model.mean_, pca_model.components_[:k].T)     # (n,k)
    X_rec = np.dot(Z, pca_model.components_[:k]) + pca_model.mean_        # (n,d)
    return X_rec

k_values = [10, 50, 100]
rand_idx = np.random.randint(0, len(X_train_flat))
orig_img = X_train_flat[rand_idx].reshape(28, 28)

H, W, pad = 28, 28, 6
mosaic_w = (len(k_values) + 1) * W + len(k_values) * pad
mosaic = np.zeros((H, mosaic_w), dtype=np.float32)

# place original
mosaic[:, :W] = orig_img
cursor = W + pad

pca_mses = []
for k in k_values:
    X_rec_one = pca_reconstruct(pca, X_train_flat[rand_idx:rand_idx+1], k)[0]
    mse = mean_squared_error(orig_img.ravel(), X_rec_one)
    pca_mses.append({"method": "PCA", "k": k, "MSE": mse})
    mosaic[:, cursor:cursor+W] = X_rec_one.reshape(H, W)
    cursor += W + pad

mosaic_disp = (mosaic - mosaic.min()) / (mosaic.max() - mosaic.min() + 1e-8)
px.imshow(mosaic_disp, color_continuous_scale="gray", zmin=0, zmax=1,
          title=f"Original (left) and PCA Reconstructions at k={k_values}").update_coloraxes(showscale=False)\
  .update_xaxes(showticklabels=False).update_yaxes(showticklabels=False).show()

# 2b) Reconstruction error vs number of components (define missing variables)
def mse_vs_k(pca_model: PCA, X_flat: np.ndarray, ks: list[int], sample_n: int = 500) -> tuple[list[int], list[float]]:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(X_flat), size=min(sample_n, len(X_flat)), replace=False)
    Xs = X_flat[idx]
    out = []
    for k in ks:
        Xr = pca_reconstruct(pca_model, Xs, k)
        out.append(mean_squared_error(Xs, Xr))
    return ks, out

components_list = [5, 10, 20, 30, 50, 75, 100, 150, 200]
components_list, reconstruction_errors = mse_vs_k(pca, X_train_flat, components_list, sample_n=800)

px.line(x=components_list, y=reconstruction_errors, markers=True,
        title="Reconstruction Error vs PCA Components",
        labels={"x": "Number of Components", "y": "MSE"}).show()

# 3) SVD: apply and compare with PCA (by reconstruction MSE)
data_mean = X_train_flat.mean(axis=0)
Xc = X_train_flat - data_mean
svd_maxk = max(k_values)
svd = TruncatedSVD(n_components=svd_maxk, random_state=RNG_SEED)
svd.fit(Xc)

Z_full = Xc @ svd.components_.T   # (n, svd_maxk)
svd_mses = []
for k in k_values:
    Xc_rec = Z_full[:, :k] @ svd.components_[:k, :]
    X_rec  = Xc_rec + data_mean
    mse = mean_squared_error(X_train_flat[rand_idx], X_rec[rand_idx])
    svd_mses.append({"method": "SVD", "k": k, "MSE": mse})

cmp_df = pd.DataFrame(pca_mses + svd_mses)
px.bar(cmp_df, x="k", y="MSE", color="method", barmode="group",
       title="Reconstruction Error (MSE) — PCA vs SVD at k = 10, 50, 100",
       text="MSE", color_discrete_sequence=px.colors.qualitative.Set2)\
  .update_traces(texttemplate="%{text:.4f}", textposition="outside")\
  .update_layout(xaxis_title="Number of components (k)", yaxis_title="MSE").show()

# ============== Phase 3: Clustering ==============

# Choose PCA features (≈95% variance)
cum = np.cumsum(pca.explained_variance_ratio_)
N_PCS_FOR_CLUSTER = int(np.searchsorted(cum, 0.95) + 1)
X_feat = X_train_pca[:, :N_PCS_FOR_CLUSTER]   # <— now defined

# (optional) subsample for speed
SAMPLE_N = min(12000, len(X_feat))
rng = np.random.default_rng(RNG_SEED)
idx = rng.choice(len(X_feat), size=SAMPLE_N, replace=False)
Xc = X_feat[idx]

print(f"Clustering on {Xc.shape[0]} samples with {Xc.shape[1]} PCA dims (~95% variance).")

# 1) K-MEANS across K values
K_VALUES = [8, 10, 12, 15]
k_results = []
for k in K_VALUES:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED)
    labels = km.fit_predict(Xc)
    inertia = km.inertia_
    sil = silhouette_score(Xc, labels) if len(np.unique(labels)) > 1 else np.nan
    k_results.append({"K": k, "inertia": inertia, "silhouette": sil})

kdf = pd.DataFrame(k_results)
px.line(kdf, x="K", y="inertia", markers=True, title="K-Means — Inertia vs K")\
  .update_layout(xaxis_title="K", yaxis_title="Inertia (lower is better)").show()
px.line(kdf, x="K", y="silhouette", markers=True, title="K-Means — Silhouette Score vs K")\
  .update_layout(xaxis_title="K", yaxis_title="Silhouette (higher is better)").show()

# 2) DBSCAN grid
EPS_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
MIN_SAMPLES_VALUES = [5, 10, 15]

db_rows = []
for eps in EPS_VALUES:
    for ms in MIN_SAMPLES_VALUES:
        db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
        labels = db.fit_predict(Xc)
        n_clusters = len(set(labels) - {-1})
        noise_ratio = float(np.mean(labels == -1))
        if n_clusters >= 2:
            # silhouette on non-noise points only
            mask = labels != -1
            sil = silhouette_score(Xc[mask], labels[mask]) if mask.any() else np.nan
        else:
            sil = np.nan
        db_rows.append({
            "eps": eps,
            "min_samples": ms,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette": sil
        })

dbdf = pd.DataFrame(db_rows)

# Heatmap of silhouette
heat = dbdf.pivot(index="min_samples", columns="eps", values="silhouette")
go.Figure(data=go.Heatmap(
    z=heat.values,
    x=heat.columns.astype(str),
    y=heat.index.astype(str),
    colorbar=dict(title="Silhouette"),
    zmin=0, zmax=1
)).update_layout(
    title="DBSCAN — Silhouette over (eps, min_samples)",
    xaxis_title="eps", yaxis_title="min_samples"
).show()

# Noise ratio chart (we'll highlight after we pick best_db)
px.bar(dbdf, x="eps", y="noise_ratio", color="min_samples", barmode="group",
       title="DBSCAN — Noise ratio by params",
       labels={"noise_ratio": "Noise ratio"}).show()

# 3) Best selection (instructor criteria)
best_kmeans = kdf.sort_values(by=["silhouette", "inertia"], ascending=[False, True]).iloc[0]

MAX_NOISE_RATIO = 0.80
db_candidates = dbdf[
    dbdf["silhouette"].notna() &
    (dbdf["n_clusters"] >= 2) &
    (dbdf["noise_ratio"] <= MAX_NOISE_RATIO)
]
if len(db_candidates):
    best_db = db_candidates.sort_values(
        by=["silhouette", "noise_ratio", "n_clusters"],
        ascending=[False, True, False]
    ).iloc[0]
    selection_note = f"(DBSCAN picked by silhouette; noise ≤ {int(MAX_NOISE_RATIO*100)}% filter applied)"
else:
    best_db = dbdf[dbdf["silhouette"].notna()]\
        .sort_values(by=["silhouette", "noise_ratio", "n_clusters"],
                     ascending=[False, True, False]).iloc[0]
    selection_note = "(DBSCAN fallback: no params passed noise filter — chose highest silhouette overall)"

print("\nBest settings (by Silhouette; inertia as tiebreaker for K-Means)")
print(f"K-Means: K={int(best_kmeans.K)}, silhouette={best_kmeans.silhouette:.4f}, inertia={best_kmeans.inertia:.0f}")

print("\nBest settings (DBSCAN)")
print(selection_note)
print(f"eps={best_db.eps}, min_samples={int(best_db.min_samples)}, "
      f"clusters={int(best_db.n_clusters)}, noise={best_db.noise_ratio:.2%}, "
      f"silhouette={best_db.silhouette:.4f}")

# ============== Phase 4: Analysis & Interpretation ==============

# Guards for selected settings
K_best = int(best_kmeans.K)
db_eps, db_min = float(best_db.eps), int(best_db.min_samples)

# PCA feature spaces
PC2 = X_train_pca[:, :2]
PC3 = X_train_pca[:, :3]

# Fit on full data with chosen params
km_final = KMeans(n_clusters=K_best, n_init=20, random_state=RNG_SEED)
km_labels = km_final.fit_predict(X_feat)

db_final = DBSCAN(eps=db_eps, min_samples=db_min, n_jobs=-1)
db_labels = db_final.fit_predict(X_feat)  # -1 = noise

# Visualize in PCA 2D/3D
SAMPLE_N = min(10000, len(PC2))
idx = np.random.default_rng(RNG_SEED).choice(len(PC2), size=SAMPLE_N, replace=False)

px.scatter(x=PC2[idx, 0], y=PC2[idx, 1],
           color=pd.Series(km_labels[idx], dtype="category").astype(str),
           title=f"K-Means (K={K_best}) — PCA 2D projection",
           labels={"x":"PC1","y":"PC2","color":"cluster"}).show()

px.scatter_3d(x=PC3[idx, 0], y=PC3[idx, 1], z=PC3[idx, 2],
              color=pd.Series(km_labels[idx], dtype="category").astype(str),
              title=f"K-Means (K={K_best}) — PCA 3D projection")\
  .update_traces(marker=dict(size=3)).show()

px.scatter(x=PC2[idx, 0], y=PC2[idx, 1],
           color=pd.Series(db_labels[idx], dtype="category").astype(str),
           title=f"DBSCAN (eps={db_eps}, min_samples={db_min}) — PCA 2D projection",
           labels={"x":"PC1","y":"PC2","color":"cluster"}).show()

px.scatter_3d(x=PC3[idx, 0], y=PC3[idx, 1], z=PC3[idx, 2],
              color=pd.Series(db_labels[idx], dtype="category").astype(str),
              title=f"DBSCAN (eps={db_eps}, min_samples={db_min}) — PCA 3D projection")\
  .update_traces(marker=dict(size=3)).show()

# 2) Compare clusters to true labels — stacked bars (concise)
def stacked_split(true_y, labels, algo_name, top_n=3):
    """
    For each true class, show a stacked bar of its top-N clusters (row-normalized).
    Remaining clusters are collapsed into 'other'. DBSCAN noise (-1) is labeled 'noise'.
    """
    df = pd.DataFrame({"y": true_y, "c": labels})
    rows = []
    for cls in range(len(CLASS_NAMES)):
        sub = df[df["y"] == cls]["c"].value_counts(normalize=True).sort_values(ascending=False)
        parts = []
        for cid, frac in sub.items():
            lab = "noise" if cid == -1 else f"c{int(cid)}"
            parts.append((lab, float(frac)))
        parts.sort(key=lambda t: t[1], reverse=True)
        top = parts[:top_n]
        other_frac = max(0.0, 1.0 - sum(f for _, f in top))
        for lab, frac in top:
            rows.append({"class": CLASS_NAMES[cls], "cluster": lab, "fraction": frac})
        if other_frac > 1e-6:
            rows.append({"class": CLASS_NAMES[cls], "cluster": "other", "fraction": other_frac})

    plot_df = pd.DataFrame(rows)
    fig = px.bar(plot_df, x="class", y="fraction", color="cluster",
                 barmode="stack",
                 title=f"{algo_name} — per-class split across top {top_n} clusters",
                 labels={"fraction": "Fraction (row-normalized)"})
    fig.update_layout(xaxis_title="True class", yaxis_title="Fraction")
    fig.show()
    return plot_df

_ = stacked_split(y_train, km_labels,  f"K-Means (K={K_best})", top_n=3)
_ = stacked_split(y_train, db_labels,  f"DBSCAN (eps={db_eps}, min_samples={db_min})", top_n=3)

# 3) Class purity
def class_purity(y_true, labels, ignore=-1):
    df = pd.DataFrame({"y": y_true, "c": labels})
    if ignore is not None:
        df = df[df["c"] != ignore]
    out = []
    for cls in range(len(CLASS_NAMES)):
        sub = df[df["y"] == cls]["c"].value_counts()
        n = sub.sum()
        purity = (sub.max() / n) if n > 0 else np.nan
        best_c = sub.idxmax() if n > 0 else None
        out.append({"class": CLASS_NAMES[cls], "purity": purity, "best_cluster": best_c})
    return pd.DataFrame(out)

pur_km = class_purity(y_train, km_labels, ignore=None)
pur_db = class_purity(y_train, db_labels, ignore=-1)

px.bar(pur_km, x="class", y="purity", title="Class purity — K-Means",
       text=pur_km["purity"].map(lambda v: f"{v:.2f}"))\
  .update_layout(yaxis_title="Purity (0–1)").show()

px.bar(pur_db, x="class", y="purity", title="Class purity — DBSCAN (noise ignored)",
       text=pur_db["purity"].map(lambda v: f"{v:.2f}"))\
  .update_layout(yaxis_title="Purity (0–1)").show()

# 4) Cluster quality (silhouette)
def silhouette_stats(X, labels, title):
    # For DBSCAN, exclude noise (-1) for silhouette
    unique = np.unique(labels)
    mask = (labels != -1) if (-1 in unique) else np.ones_like(labels, dtype=bool)
    X_use, L_use = X[mask], labels[mask]
    if (len(L_use) == 0) or (len(np.unique(L_use)) < 2):
        print(f"{title}: not enough non-noise clusters for silhouette.")
        return np.nan
    s_global = silhouette_score(X_use, L_use)
    s_each = silhouette_samples(X_use, L_use)
    df = pd.DataFrame({"cluster": L_use, "sil": s_each})
    per = df.groupby("cluster")["sil"].mean().reset_index().sort_values("sil", ascending=False)
    print(f"{title} — global silhouette = {s_global:.4f}")
    px.bar(per, x="cluster", y="sil",
           title=f"{title} — mean silhouette per cluster",
           text=per["sil"].map(lambda v: f"{v:.3f}"),
           labels={"sil": "Mean silhouette"}).show()
    return s_global

sil_km = silhouette_stats(X_feat, km_labels, f"K-Means (K={K_best})")
sil_db = silhouette_stats(X_feat, db_labels, f"DBSCAN (eps={db_eps}, min_samples={db_min})")

print("\nPhase 4 complete ✅ — Visual comparisons & metrics rendered.")
