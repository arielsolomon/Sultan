"""
Clustering Analysis: KMeans & DBSCAN on Feature Vectors
Detects the outlier vector and measures its distance from cluster centers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive, save-only — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG — adjust these to match your setup
# ─────────────────────────────────────────────
VECTORS_FOLDER = "/home/user1/ariel/Sultan/data/feature_vec_1/individual_features_10/"
N_CLUSTERS_KMEANS = 1
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 2
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 1. LOAD VECTORS
# ─────────────────────────────────────────────
def load_vectors(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in '{folder}'")

    vectors, names = [], []
    for fname in files:
        vec = np.load(os.path.join(folder, fname))
        vectors.append(vec.flatten())
        names.append(os.path.splitext(fname)[0])

    print(f"Loaded {len(vectors)} vectors: {names}")
    return np.array(vectors), names


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(2, X_scaled.shape[1]))
    X_2d = pca.fit_transform(X_scaled)
    return X_scaled, X_2d


# ─────────────────────────────────────────────
# 3. KMEANS
# ─────────────────────────────────────────────
def run_kmeans(X_scaled, names):
    print("\n" + "═" * 50)
    print("  K-MEANS CLUSTERING")
    print("═" * 50)

    km = KMeans(n_clusters=N_CLUSTERS_KMEANS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_

    distances = np.array([
        np.linalg.norm(X_scaled[i] - centroids[labels[i]])
        for i in range(len(X_scaled))
    ])

    print(f"\n{'Vector':<30} {'Cluster':>8} {'Dist to Centroid':>18}")
    print("-" * 60)
    for name, label, dist in zip(names, labels, distances):
        print(f"{name:<30} {label:>8}       {dist:>12.4f}")

    outlier_idx = np.argmax(distances)
    print(f"\n⚠  Most distant vector: '{names[outlier_idx]}'"
          f"  (distance = {distances[outlier_idx]:.4f})")

    return labels, centroids, distances, outlier_idx


# ─────────────────────────────────────────────
# 4. DBSCAN
# ─────────────────────────────────────────────
def run_dbscan(X_scaled, names):
    print("\n" + "═" * 50)
    print("  DBSCAN CLUSTERING")
    print("═" * 50)

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(X_scaled)

    core_labels = [l for l in set(labels) if l != -1]
    distances = np.full(len(X_scaled), np.nan)

    if core_labels:
        centroids = {
            l: X_scaled[labels == l].mean(axis=0) for l in core_labels
        }
        for i, label in enumerate(labels):
            if label == -1:
                distances[i] = min(
                    np.linalg.norm(X_scaled[i] - c) for c in centroids.values()
                )
            else:
                distances[i] = np.linalg.norm(X_scaled[i] - centroids[label])
    else:
        pseudo_centroid = X_scaled.mean(axis=0)
        for i in range(len(X_scaled)):
            distances[i] = np.linalg.norm(X_scaled[i] - pseudo_centroid)
        print("  ⚠ All points classified as noise. Using global centroid.")

    print(f"\n{'Vector':<30} {'Cluster':>8} {'Dist to Cluster':>18}")
    print("-" * 60)
    for name, label, dist in zip(names, labels, distances):
        label_str = "NOISE" if label == -1 else str(label)
        print(f"{name:<30} {label_str:>8}       {dist:>12.4f}")

    outlier_idx = np.nanargmax(distances)
    print(f"\n⚠  Most distant vector: '{names[outlier_idx]}'"
          f"  (distance = {distances[outlier_idx]:.4f})")

    return labels, distances, outlier_idx


# ─────────────────────────────────────────────
# 5. VISUALIZATION (save only)
# ─────────────────────────────────────────────
PALETTE = [
    "#4C9BE8", "#F28C38", "#6BCB77", "#D65DB1",
    "#FF6B6B", "#845EC2", "#00C9A7", "#FFC75F"
]

def plot_results(X_2d, names,
                 km_labels, km_distances, km_outlier,
                 db_labels, db_distances, db_outlier):

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0F1117")
    for ax in axes:
        ax.set_facecolor("#1A1D2E")
        ax.tick_params(colors="#AAAAAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    def scatter_panel(ax, labels, distances, outlier_idx, title, centroids_2d=None):
        unique_labels = sorted(set(labels))
        label_colors = {
            l: "#888888" if l == -1 else PALETTE[i % len(PALETTE)]
            for i, l in enumerate(unique_labels)
        }

        for i, (x, y) in enumerate(X_2d):
            color = label_colors[labels[i]]
            size = 180 if i == outlier_idx else 100
            edge = "#FF4444" if i == outlier_idx else "#FFFFFF"
            edgewidth = 2.5 if i == outlier_idx else 0.8
            ax.scatter(x, y, c=color, s=size,
                       edgecolors=edge, linewidths=edgewidth, zorder=3)
            ax.annotate(
                names[i], (x, y),
                textcoords="offset points", xytext=(6, 6),
                fontsize=3.5, color="#DDDDDD",
                bbox=dict(boxstyle="round,pad=0.2", fc="#1A1D2E", alpha=0.7, ec="none")
            )

        if centroids_2d is not None:
            for c2d in centroids_2d:
                ax.scatter(*c2d, marker="X", s=250, c="#FFD700",
                           edgecolors="#FFFFFF", linewidths=1.5, zorder=5,
                           label="Centroid")

        for i, (x, y) in enumerate(X_2d):
            d_str = f"{distances[i]:.3f}" if not np.isnan(distances[i]) else "N/A"
            ax.annotate(
                d_str, (x, y),
                textcoords="offset points", xytext=(6, -14),
                fontsize=3.5, color="#AAAAFF", style="italic"
            )

        ax.set_title(title, color="#FFFFFF", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("PCA Component 1", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("PCA Component 2", color="#AAAAAA", fontsize=9)

        patches = [
            mpatches.Patch(
                color=label_colors[l],
                label="Noise" if l == -1 else f"Cluster {l}"
            )
            for l in unique_labels
        ]
        ax.legend(handles=patches, framealpha=0.3, labelcolor="#DDDDDD",
                  facecolor="#0F1117", edgecolor="#444466", fontsize=8)

    km_centroids_2d = np.array([
        X_2d[km_labels == l].mean(axis=0)
        for l in sorted(set(km_labels))
    ])

    scatter_panel(axes[0], km_labels, km_distances, km_outlier,
                  "K-Means Clustering", km_centroids_2d)
    scatter_panel(axes[1], db_labels, db_distances, db_outlier,
                  "DBSCAN Clustering")

    fig.suptitle(
        "Feature Vector Clustering — Outlier Detection",
        color="#FFFFFF", fontsize=15, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    out_path = os.path.join(VECTORS_FOLDER, "clustering_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n📊 Plot saved → {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    X_raw, names = load_vectors(VECTORS_FOLDER)
    X_scaled, X_2d = preprocess(X_raw)

    km_labels, km_centroids, km_distances, km_outlier = run_kmeans(X_scaled, names)
    db_labels, db_distances, db_outlier = run_dbscan(X_scaled, names)

    plot_results(
        X_2d, names,
        km_labels, km_distances, km_outlier,
        db_labels, db_distances, db_outlier
    )
