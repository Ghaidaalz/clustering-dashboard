# app.py
# Streamlit dashboard for Fashion-MNIST — Phases 1–6

import warnings
warnings.filterwarnings("ignore")

import io
import random
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error
from sklearn.datasets import fetch_openml

# ----------------------------- performance caps (Cloud-friendly) -----------------------------
MAX_PLOT_POINTS = 8000       # for 2D/3D plots
MAX_SILH_POINTS = 8000       # for silhouette computations
MAX_POOL_POINTS = 8000       # for similarity search pool in Phase 6

# ---------------------------------------------------------------------
# Page config + simple dark styling (works with the default dark theme)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Fashion-MNIST – Clustering & PCA",
    layout="wide",
    initial_sidebar_state="expanded"
)

PLOTLY_TMPL = "plotly_dark"
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    .streamlit-expanderHeader {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# Globals / constants
# ---------------------------------------------------------------------
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# =============================== Helpers for Phase 6 (upload & similarity) ===============================
def load_and_prepare_image(file_bytes_or_pil, invert=False):
    """
    Reads an uploaded image or PIL.Image, converts to 28x28 grayscale,
    returns:
      - img_28x28: (28, 28) float32 in [0,1]
      - flat: (784,) float32 flattened and normalized
    """
    if isinstance(file_bytes_or_pil, Image.Image):
        pil = file_bytes_or_pil
    else:
        pil = Image.open(io.BytesIO(file_bytes_or_pil))

    pil = pil.convert("L").resize((28, 28))  # grayscale 28x28
    arr = np.array(pil).astype(np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    flat = arr.reshape(-1).astype(np.float32)
    return arr, flat


def topk_similar(feature_vec, features_matrix, pool_idx, k=6):
    """
    Returns indices (within pool_idx) of the k most similar items to feature_vec
    using Euclidean distance in the PCA feature space.
    """
    pool_feats = features_matrix[pool_idx]            # (n_pool, n_features)
    d2 = np.sum((pool_feats - feature_vec) ** 2, axis=1)
    order = np.argsort(d2)[:k]
    return pool_idx[order], np.sqrt(d2[order])


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """
    Fetch Fashion-MNIST from OpenML (no Keras/TF needed).
    Returns the same shapes/types as before:
      (X_train, y_train), (X_test, y_test) but flattened to 4 arrays.
    """
    X, y = fetch_openml("Fashion-MNIST", version=1, as_frame=False, return_X_y=True)
    X = X.reshape(-1, 28, 28).astype("uint8")
    try:
        y = y.astype("int64").to_numpy()
    except AttributeError:
        y = y.astype("int64")

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


@st.cache_data(show_spinner=False)
def normalize_and_flatten(X_train, X_test):
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_test_norm  = X_test.astype(np.float32) / 255.0
    X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
    X_test_flat  = X_test_norm.reshape(len(X_test_norm), -1)
    return X_train_norm, X_test_norm, X_train_flat, X_test_flat


@st.cache_data(show_spinner=False)
def fit_pca(X_flat):
    pca = PCA()
    pca.fit(X_flat)
    return pca


@st.cache_data(show_spinner=False)
def pca_projection(X_flat, n_components):
    p = PCA(n_components=n_components)
    Z = p.fit_transform(X_flat)
    return Z


@st.cache_data(show_spinner=False)
def fit_svd(X_centered, n_components):
    svd = TruncatedSVD(n_components=n_components, random_state=RNG_SEED)
    svd.fit(X_centered)
    return svd


# ---------------------------------------------------------------------
# Small utilities (plots & metrics)
# ---------------------------------------------------------------------
def pca_reconstruct(pca_model, X_flat, k):
    """Manual project/back-project with first k components."""
    Z = np.dot(X_flat - pca_model.mean_, pca_model.components_[:k].T)
    X_rec = np.dot(Z, pca_model.components_[:k]) + pca_model.mean_
    return X_rec


def stacked_split(true_y, labels, top_n=3, algo_title=""):
    """
    For each true class, show a stacked bar of its top-N clusters (row-normalized).
    Remaining clusters are collapsed into 'other'. DBSCAN noise (-1) is kept as 'noise'.
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
    fig = px.bar(
        plot_df, x="class", y="fraction", color="cluster",
        barmode="stack",
        title=f"{algo_title} — per-class split across top {top_n} clusters",
        template=PLOTLY_TMPL, labels={"fraction": "Fraction (row-normalized)"}
    )
    fig.update_layout(xaxis_title="True class", yaxis_title="Fraction")
    st.plotly_chart(fig, use_container_width=True)
    return plot_df


def silhouette_per_cluster(X, labels, title="", bar_color="#FFB6C1", max_n=None, seed=42):
    unique_labels = np.unique(labels)
    # need >=2 labels and at least one non-noise point
    if len(unique_labels) < 2 or (len(unique_labels) == 1 and unique_labels[0] == -1):
        return None, None

    if max_n is not None and len(X) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_n, replace=False)
        X_use = X[idx]
        L_use = labels[idx]
        if len(np.unique(L_use)) < 2:
            return None, None
    else:
        X_use = X
        L_use = labels

    sil_vals = silhouette_samples(X_use, L_use)
    df_sil = pd.DataFrame({"cluster": L_use, "silhouette": sil_vals})
    df_mean = df_sil.groupby('cluster', as_index=False)['silhouette'].mean()

    fig = px.bar(df_mean, x='cluster', y='silhouette', text='silhouette', title=title)
    fig.update_traces(marker_color=bar_color, texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_title="Mean silhouette", xaxis_title="cluster")

    global_sil = float(df_sil['silhouette'].mean())
    return fig, global_sil


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Settings")

with st.sidebar.expander("Data & Sampling", expanded=True):
    sample_for_clustering = st.number_input(
        "Clustering sample size (for speed)", min_value=1000, max_value=60000, step=1000, value=12000
    )

with st.sidebar.expander("K-Means", expanded=True):
    k_values = st.multiselect("K candidates", [8, 10, 12, 15], default=[8, 10, 12, 15])
    k_final = st.number_input("Final K for visualization", min_value=2, max_value=20, value=8)

with st.sidebar.expander("DBSCAN", expanded=True):
    eps_val = st.select_slider("eps", options=[1.0, 1.5, 2.0, 2.5, 3.0], value=3.0)
    ms_val = st.select_slider("min_samples", options=[5, 10, 15], value=5)
    max_noise_ratio = st.slider("Noise cap for 'best' selection", 0.0, 1.0, 0.80, 0.05)

st.sidebar.markdown("---")
_ = st.sidebar.button("Run / Re-compute")


# ---------------------------------------------------------------------
# Phase 1 – Data understanding & preprocessing
# ---------------------------------------------------------------------
st.title("Fashion-MNIST — PCA, SVD & Clustering Dashboard")

with st.expander("Phase 1 — Data Understanding & Preprocessing", expanded=True):
    X_train, y_train, X_test, y_test = load_data()
    st.write(f"**Train images:** {X_train.shape}  |  **Test images:** {X_test.shape}")

    # normalization & flatten
    X_train_norm, X_test_norm, X_train_flat, X_test_flat = normalize_and_flatten(X_train, X_test)

    # distributions / histograms
    max_images_for_hist = 4000
    sample_idxs = np.random.choice(len(X_train), size=min(max_images_for_hist, len(X_train)), replace=False)

    sample_pixels_raw = X_train[sample_idxs].reshape(-1)
    fig = px.histogram(sample_pixels_raw, nbins=30,
                       title="Raw pixel intensity histogram (sampled)",
                       color_discrete_sequence=['#FF7F50'], template=PLOTLY_TMPL)
    fig.update_layout(xaxis_title="Pixel value (0–255)", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

    sample_pixels_norm = X_train_norm[sample_idxs].reshape(-1)
    fig = px.histogram(sample_pixels_norm, nbins=30,
                       title="Pixel intensity histogram after normalization (sampled)",
                       color_discrete_sequence=['#6A5ACD'], template=PLOTLY_TMPL)
    st.plotly_chart(fig, use_container_width=True)

    mean_img_norm = X_train_norm.mean(axis=0)
    std_img_norm = X_train_norm.std(axis=0)
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(px.imshow(mean_img_norm, color_continuous_scale="Cividis",
                                  title="Mean image (normalized)", template=PLOTLY_TMPL), use_container_width=True)
    with colB:
        st.plotly_chart(px.imshow(std_img_norm, color_continuous_scale="Inferno",
                                  title="Std-dev image (normalized)", template=PLOTLY_TMPL), use_container_width=True)

# ---------------------------------------------------------------------
# Phase 2 – PCA & SVD
# ---------------------------------------------------------------------
with st.expander("Phase 2 — PCA & SVD", expanded=True):
    pca = fit_pca(X_train_flat)
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=evr, mode='lines+markers', name='Explained Variance'))
    fig.add_trace(go.Scatter(y=cum_evr, mode='lines+markers', name='Cumulative Explained Variance'))
    fig.update_layout(title='PCA Explained Variance', xaxis_title='Components',
                      yaxis_title='Variance Ratio', template=PLOTLY_TMPL)
    st.plotly_chart(fig, use_container_width=True)

    # reconstruction example at k = 10, 50, 100
    k_values_recon = [10, 50, 100]
    rand_idx = np.random.randint(0, len(X_train_flat))
    orig_img = X_train_flat[rand_idx].reshape(28, 28)

    H, W, pad = 28, 28, 6
    mosaic_w = (len(k_values_recon) + 1) * W + len(k_values_recon) * pad
    mosaic = np.zeros((H, mosaic_w), dtype=np.float32)
    mosaic[:, :W] = orig_img
    cursor = W + pad

    pca_mses = []
    for k in k_values_recon:
        X_rec_one = pca_reconstruct(pca, X_train_flat[rand_idx:rand_idx+1], k)[0]
        mse = mean_squared_error(orig_img.ravel(), X_rec_one)
        pca_mses.append({"method": "PCA", "k": k, "MSE": mse})
        mosaic[:, cursor:cursor+W] = X_rec_one.reshape(H, W)
        cursor += W + pad

    mosaic_disp = (mosaic - mosaic.min()) / (mosaic.max() - mosaic.min() + 1e-8)
    st.plotly_chart(px.imshow(mosaic_disp, color_continuous_scale="gray", zmin=0, zmax=1,
                              title=f"Original (left) and PCA Reconstructions at k={k_values_recon}",
                              template=PLOTLY_TMPL), use_container_width=True)

    # PCA 2D/3D projections
    PC2 = pca_projection(X_train_flat, 2)
    PC3 = pca_projection(X_train_flat, 3)

    st.plotly_chart(
        px.scatter(x=PC2[:, 0], y=PC2[:, 1],
                   color=[CLASS_NAMES[i] for i in y_train],
                   title="PCA 2D Projection", labels={'x': 'PC1', 'y': 'PC2'},
                   template=PLOTLY_TMPL),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter_3d(x=PC3[:, 0], y=PC3[:, 1], z=PC3[:, 2],
                      color=[CLASS_NAMES[i] for i in y_train],
                      title="PCA 3D Projection", template=PLOTLY_TMPL),
        use_container_width=True
    )

    # SVD vs PCA recon error at k
    data_mean = X_train_flat.mean(axis=0)
    Xc = X_train_flat - data_mean
    svd = fit_svd(Xc, max(k_values_recon))
    Z_full = Xc @ svd.components_.T

    svd_mses = []
    for k in k_values_recon:
        Xc_rec = Z_full[:, :k] @ svd.components_[:k, :]
        X_rec = Xc_rec + data_mean
        mse = mean_squared_error(X_train_flat[rand_idx], X_rec[rand_idx])
        svd_mses.append({"method": "SVD", "k": k, "MSE": mse})

    cmp_df = pd.DataFrame(pca_mses + svd_mses)
    fig = px.bar(cmp_df, x="k", y="MSE", color="method", barmode="group",
                 title="Reconstruction Error (MSE) — PCA vs SVD (k=10,50,100)",
                 text="MSE", color_discrete_sequence=px.colors.qualitative.Set2,
                 template=PLOTLY_TMPL)
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(xaxis_title="k", yaxis_title="MSE")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Phase 3 – Clustering
# ---------------------------------------------------------------------
with st.expander("Phase 3 — Clustering", expanded=True):
    # choose ~95% variance PCA features
    cum = np.cumsum(fit_pca(X_train_flat).explained_variance_ratio_)
    N_PCS_FOR_CLUSTER = int(np.searchsorted(cum, 0.95) + 1)
    X_feat = pca_projection(X_train_flat, N_PCS_FOR_CLUSTER)

    # sample for clustering speed
    N = min(sample_for_clustering, len(X_feat))
    idx = np.random.default_rng(RNG_SEED).choice(len(X_feat), size=N, replace=False)
    Xc_small = X_feat[idx]

    st.write(
        f"Clustering on **{Xc_small.shape[0]}** samples with **{Xc_small.shape[1]}** PCA dims (~95% variance)."
    )

    # 1) KMeans grid
    k_results = []
    for k in (k_values or [8, 10, 12, 15]):
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED)
        labels = km.fit_predict(Xc_small)
        inertia = km.inertia_
        sil = silhouette_score(Xc_small, labels) if len(np.unique(labels)) > 1 else np.nan
        k_results.append({"K": k, "inertia": inertia, "silhouette": sil})
    kdf = pd.DataFrame(k_results)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(kdf, x="K", y="inertia", markers=True, title="K-Means — Inertia vs K",
                                template=PLOTLY_TMPL), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(kdf, x="K", y="silhouette", markers=True, title="K-Means — Silhouette vs K",
                                template=PLOTLY_TMPL), use_container_width=True)

    # 2) DBSCAN grid
    EPS_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
    MIN_SAMPLES_VALUES = [5, 10, 15]
    db_rows = []
    for eps in EPS_VALUES:
        for ms in MIN_SAMPLES_VALUES:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            lab = db.fit_predict(Xc_small)
            n_clusters = len(set(lab) - {-1})
            noise_ratio = float(np.mean(lab == -1))
            if n_clusters >= 2:
                sil = silhouette_score(Xc_small[lab != -1], lab[lab != -1])
            else:
                sil = np.nan
            db_rows.append({"eps": eps, "min_samples": ms, "n_clusters": n_clusters,
                            "noise_ratio": noise_ratio, "silhouette": sil})
    dbdf = pd.DataFrame(db_rows)

    heat = dbdf.pivot(index="min_samples", columns="eps", values="silhouette")
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.values, x=heat.columns.astype(str), y=heat.index.astype(str),
            colorbar=dict(title="Silhouette"), zmin=0, zmax=1, colorscale="Viridis"
        )
    )
    fig.update_layout(title="DBSCAN — Silhouette over (eps, min_samples)",
                      xaxis_title="eps", yaxis_title="min_samples", template=PLOTLY_TMPL)
    st.plotly_chart(fig, use_container_width=True)

    # pick "best" by silhouette with a noise cap
    db_candidates = dbdf[
        dbdf["silhouette"].notna() &
        (dbdf["n_clusters"] >= 2) &
        (dbdf["noise_ratio"] <= max_noise_ratio)
    ]

    if len(db_candidates):
        best_db = (
            db_candidates.sort_values(
                by=["silhouette", "noise_ratio", "n_clusters"],
                ascending=[False, True, False]
            ).iloc[0]
        )
        selection_note = f"(DBSCAN picked by silhouette; noise ≤ {int(max_noise_ratio * 100)}% filter applied)"
    else:
        best_db = (
            dbdf[dbdf["silhouette"].notna()]
            .sort_values(by=["silhouette", "noise_ratio", "n_clusters"],
                         ascending=[False, True, False]).iloc[0]
        )
        selection_note = "(DBSCAN fallback: no params passed noise cap — chose highest silhouette overall)"

    # concise summary block
    best_kmeans = (kdf.sort_values(by=["silhouette", "inertia"], ascending=[False, True]).iloc[0])
    st.code(
f"""Best settings (by Silhouette; inertia as tie-breaker for K-Means)
K-Means: K={int(best_kmeans.K)}, silhouette={best_kmeans.silhouette:.4f}, inertia={best_kmeans.inertia:.0f}

Best settings (DBSCAN)
{selection_note}
eps={best_db.eps}, min_samples={int(best_db.min_samples)}, clusters={int(best_db.n_clusters)},
noise={best_db.noise_ratio:.2%}, silhouette={best_db.silhouette:.4f}
"""
    )

# ---------------------------------------------------------------------
# Phase 4 – Analysis & interpretation (visuals & metrics)
# ---------------------------------------------------------------------
with st.expander("Phase 4 — Analysis & Interpretation", expanded=True):
    cum = np.cumsum(fit_pca(X_train_flat).explained_variance_ratio_)
    N_PCS_FOR_CLUSTER = int(np.searchsorted(cum, 0.95) + 1)
    X_feat = pca_projection(X_train_flat, N_PCS_FOR_CLUSTER)
    PC2_full = X_feat[:, :2]
    PC3_full = X_feat[:, :3]

    km_final = KMeans(n_clusters=int(k_final), n_init=20, random_state=RNG_SEED)
    km_labels = km_final.fit_predict(X_feat)

    db_final = DBSCAN(eps=float(eps_val), min_samples=int(ms_val), n_jobs=-1)
    db_labels = db_final.fit_predict(X_feat)

    # ---------- NEW: sample once for plotting & per-class bars ----------
    rng = np.random.default_rng(RNG_SEED)
    plot_n = min(MAX_PLOT_POINTS, len(PC2_full))
    plot_idx = rng.choice(len(PC2_full), size=plot_n, replace=False)

    # For Phase 6 (similarity), keep a full index reference
    idx_all = np.arange(len(X_train_flat))

    # 2D/3D plots (sampled)
    st.plotly_chart(
        px.scatter(x=PC2_full[plot_idx, 0], y=PC2_full[plot_idx, 1],
                   color=pd.Series(km_labels[plot_idx], dtype="category").astype(str),
                   title=f"K-Means (K={int(k_final)}) — PCA 2D projection",
                   labels={"x": "PC1", "y": "PC2", "color": "cluster"},
                   template=PLOTLY_TMPL),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter_3d(x=PC3_full[plot_idx, 0], y=PC3_full[plot_idx, 1], z=PC3_full[plot_idx, 2],
                      color=pd.Series(km_labels[plot_idx], dtype="category").astype(str),
                      title=f"K-Means (K={int(k_final)}) — PCA 3D projection",
                      template=PLOTLY_TMPL).update_traces(marker=dict(size=3)),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(x=PC2_full[plot_idx, 0], y=PC2_full[plot_idx, 1],
                   color=pd.Series(db_labels[plot_idx], dtype="category").astype(str),
                   title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)}) — PCA 2D projection",
                   labels={"x": "PC1", "y": "PC2", "color": "cluster"},
                   template=PLOTLY_TMPL),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter_3d(x=PC3_full[plot_idx, 0], y=PC3_full[plot_idx, 1], z=PC3_full[plot_idx, 2],
                      color=pd.Series(db_labels[plot_idx], dtype="category").astype(str),
                      title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)}) — PCA 3D projection",
                      template=PLOTLY_TMPL).update_traces(marker=dict(size=3)),
        use_container_width=True
    )

    # concise per-class split bars (use the same sample so it’s fast)
    st.subheader("Per-class split across top clusters")
    stacked_split(y_train[plot_idx], km_labels[plot_idx],
                  top_n=3, algo_title=f"K-Means (K={int(k_final)})")
    stacked_split(y_train[plot_idx], db_labels[plot_idx],
                  top_n=3, algo_title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)})")

    # silhouette quality (sampled)
    st.subheader("Silhouette quality")
    fig_sk, glob_k = silhouette_per_cluster(
        X_feat, km_labels,
        title=f"K-Means (K={int(k_final)})",
        bar_color="#FFB6C1",
        max_n=MAX_SILH_POINTS
    )
    if fig_sk is not None:
        st.write(f"K-Means global silhouette: {glob_k:.4f}")
        st.plotly_chart(fig_sk, use_container_width=True)
    else:
        st.info("K-Means: not enough clusters for silhouette.")

    fig_sd, glob_d = silhouette_per_cluster(
        X_feat, db_labels,
        title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)})",
        bar_color="#ADD8E6",
        max_n=MAX_SILH_POINTS
    )
    if fig_sd is not None:
        st.write(f"DBSCAN global silhouette: {glob_d:.4f}")
        st.plotly_chart(fig_sd, use_container_width=True)
    else:
        st.info("DBSCAN: not enough non-noise clusters for silhouette.")

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Phase 6 — Upload / Pick a sample (Predict + Similarity + PCA highlight)
# ---------------------------------------------------------------------
with st.expander("Phase 6 — Upload or pick a sample (predict & similar)", expanded=True):

    # 1) Make sure we have data + PCA features (compute if needed)
    X_train, y_train, X_test, y_test = load_data()
    X_train_norm, X_test_norm, X_train_flat, X_test_flat = normalize_and_flatten(X_train, X_test)

    if 'pca' not in st.session_state:
        st.session_state.pca = fit_pca(X_train_flat)
    pca = st.session_state.pca

    if 'X_feat' not in st.session_state:
        st.session_state.X_feat = pca.transform(X_train_flat)  # PCA features for train set
    X_feat = st.session_state.X_feat
    PC2_full = X_feat[:, :2]  # for plotting

    # KMeans model (use current k_final from sidebar)
    if 'km_final' not in st.session_state or (st.session_state.k_used if 'k_used' in st.session_state else None) != int(k_final):
        st.session_state.km_final = KMeans(n_clusters=int(k_final), n_init=20, random_state=RNG_SEED).fit(X_feat)
        st.session_state.k_used = int(k_final)
    km_final = st.session_state.km_final
    km_labels = km_final.labels_

    # 2) UI: upload OR pick a sample
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        upl = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    with c2:
        pick_sample = st.checkbox("Pick dataset sample instead", value=False)
    with c3:
        invert_opt = st.checkbox("Invert colors (if needed)", value=False)

    sample_idx = st.number_input("Sample index (0..59999)", min_value=0, max_value=59999, value=0, step=1)

    # 3) Read user image -> (28x28) + flat
    user_img = None
    user_flat = None
    user_src = None

    if pick_sample:
        user_img = X_train_norm[sample_idx]                 # (28,28) in [0,1]
        user_flat = user_img.reshape(-1).astype(np.float32) # (784,)
        user_src = f"Dataset sample #{sample_idx}"
    elif upl is not None:
        user_img, user_flat = load_and_prepare_image(upl.read(), invert=invert_opt)
        user_src = "Uploaded image"
    else:
        st.info("Upload an image or tick 'Pick dataset sample instead' to continue.")

    # 4) If we have an image, predict cluster & show similar
    if user_img is not None:
        st.caption(f"Input: {user_src}")
        st.image(user_img, width=140, clamp=True)

        # Project with PCA
        user_pca = pca.transform(user_flat.reshape(1, -1))    # (1, n_pcs)

        # Predict cluster
        user_cluster = int(km_final.predict(user_pca)[0])
        st.markdown(f"**Predicted K-Means cluster:** `{user_cluster}`")

        # Similar items within same cluster (limit pool size)
        pool_idx = np.where(km_labels == user_cluster)[0]
        if len(pool_idx) > MAX_POOL_POINTS:
            rng = np.random.default_rng(RNG_SEED)
            pool_idx = rng.choice(pool_idx, size=MAX_POOL_POINTS, replace=False)

        # Find nearest neighbors in PCA space
        def _topk_similar(feature_vec, features_matrix, pool_idx, k=6):
            pool_feats = features_matrix[pool_idx]
            d2 = np.sum((pool_feats - feature_vec) ** 2, axis=1)
            order = np.argsort(d2)[:k]
            return pool_idx[order], np.sqrt(d2[order])

        nn_idx, nn_dist = _topk_similar(user_pca[0], X_feat, pool_idx, k=6)

        st.write(f"Most similar {len(nn_idx)} items from cluster {user_cluster}:")
        cols = st.columns(len(nn_idx))
        for j, (ii, dd) in enumerate(zip(nn_idx, nn_dist)):
            with cols[j]:
                st.image(X_train_norm[ii], clamp=True, width=100)
                st.caption(f"idx={ii} • dist={dd:.3f}")

        # 5) Plot PCA-2D with user point highlighted
        st.subheader("PCA 2D projection (user highlighted)")
        SUB = min(6000, len(PC2_full))
        rng = np.random.default_rng(RNG_SEED)
        sub_idx = rng.choice(len(PC2_full), size=SUB, replace=False)

        fig_user = go.Figure()
        fig_user.add_trace(go.Scatter(
            x=PC2_full[sub_idx, 0], y=PC2_full[sub_idx, 1],
            mode='markers', name='data',
            marker=dict(size=4, opacity=0.25),
            showlegend=False
        ))
        fig_user.add_trace(go.Scatter(
            x=[float(user_pca[0, 0])], y=[float(user_pca[0, 1])],
            mode='markers+text', text=["YOU"], textposition='top center',
            marker=dict(size=12, symbol='star', line=dict(width=1), color='#FFD166'),
            name='user'
        ))
        fig_user.update_layout(template=PLOTLY_TMPL, xaxis_title="PC1", yaxis_title="PC2",
                               title="User image in PCA-2D space")
        st.plotly_chart(fig_user, use_container_width=True)
