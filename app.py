# app.py
# Streamlit dashboard for Fashion-MNIST Phases 1–4

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error

from tensorflow.keras.datasets import fashion_mnist


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

# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test


@st.cache_data(show_spinner=False)
def normalize_and_flatten(X_train, X_test):
    X_train_norm = X_train.astype(np.float32) / 255.0
    X_test_norm = X_test.astype(np.float32) / 255.0
    X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
    X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)
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


import plotly.express as px

def silhouette_per_cluster(X, labels, title="", bar_color="#FFB6C1"):
    from sklearn.metrics import silhouette_samples
    import numpy as np
    import pandas as pd

    # Handle case where not enough clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None, None

    # Compute silhouette values
    sil_vals = silhouette_samples(X, labels)
    df_sil = pd.DataFrame({
        'cluster': labels,
        'silhouette': sil_vals
    })

    # Mean silhouette per cluster
    df_mean = df_sil.groupby('cluster', as_index=False)['silhouette'].mean()

    # Create bar chart
    fig = px.bar(
        df_mean,
        x='cluster',
        y='silhouette',
        text='silhouette',  # Add labels to bars
        title=title
    )

    # Customize colors & label format
    fig.update_traces(
        marker_color=bar_color,
        texttemplate='%{text:.3f}',  # Format labels to 3 decimals
        textposition='outside'       # Position above bars
    )
    fig.update_layout(
        yaxis_title="Mean silhouette",
        xaxis_title="cluster"
    )

    global_sil = df_sil['silhouette'].mean()
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

    # sampled 2D/3D plots
    N = min(10000, len(PC2_full))
    idx = np.random.default_rng(RNG_SEED).choice(len(PC2_full), size=N, replace=False)

    st.plotly_chart(
        px.scatter(x=PC2_full[idx, 0], y=PC2_full[idx, 1],
                   color=pd.Series(km_labels[idx], dtype="category").astype(str),
                   title=f"K-Means (K={int(k_final)}) — PCA 2D projection",
                   labels={"x": "PC1", "y": "PC2", "color": "cluster"},
                   template=PLOTLY_TMPL),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter_3d(x=PC3_full[idx, 0], y=PC3_full[idx, 1], z=PC3_full[idx, 2],
                      color=pd.Series(km_labels[idx], dtype="category").astype(str),
                      title=f"K-Means (K={int(k_final)}) — PCA 3D projection",
                      template=PLOTLY_TMPL).update_traces(marker=dict(size=3)),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(x=PC2_full[idx, 0], y=PC2_full[idx, 1],
                   color=pd.Series(db_labels[idx], dtype="category").astype(str),
                   title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)}) — PCA 2D projection",
                   labels={"x": "PC1", "y": "PC2", "color": "cluster"},
                   template=PLOTLY_TMPL),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter_3d(x=PC3_full[idx, 0], y=PC3_full[idx, 1], z=PC3_full[idx, 2],
                      color=pd.Series(db_labels[idx], dtype="category").astype(str),
                      title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)}) — PCA 3D projection",
                      template=PLOTLY_TMPL).update_traces(marker=dict(size=3)),
        use_container_width=True
    )

    # concise per-class split bars
    st.subheader("Per-class split across top clusters")
    stacked_split(y_train, km_labels, top_n=3, algo_title=f"K-Means (K={int(k_final)})")
    stacked_split(y_train, db_labels, top_n=3, algo_title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)})")

    # silhouette quality (custom bar colors)
    st.subheader("Silhouette quality")
    # K-Means: pastel pink
    fig_sk, glob_k = silhouette_per_cluster(
        X_feat, km_labels, title=f"K-Means (K={int(k_final)})", bar_color="#FFB6C1"
    )
    if fig_sk is not None:
        st.write(f"K-Means global silhouette: {glob_k:.4f}")
        st.plotly_chart(fig_sk, use_container_width=True)
    else:
        st.info("K-Means: not enough clusters for silhouette.")

    # DBSCAN: light blue
    fig_sd, glob_d = silhouette_per_cluster(
        X_feat, db_labels, title=f"DBSCAN (eps={float(eps_val)}, min_samples={int(ms_val)})", bar_color="#ADD8E6"
    )
    if fig_sd is not None:
        st.write(f"DBSCAN global silhouette: {glob_d:.4f}")
        st.plotly_chart(fig_sd, use_container_width=True)
    else:
        st.info("DBSCAN: not enough non-noise clusters for silhouette.")
