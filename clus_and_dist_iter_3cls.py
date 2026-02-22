"""
Clustering Analysis: KMeans & DBSCAN (3-class evaluation)

Per iteration (12 vectors total):
  - 9 randomly sampled from class-0
  - 1 randomly sampled from class-1 (known outlier)
  - 1 randomly sampled from class-0 (test_vec)
  - 1 randomly sampled from negative-control folder

Outputs:
  summary_table.csv
  summary_table_3class_dark.png
"""

import os
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FOLDER_CLASS0 = "/work/Sultan/data/feature_vec_0/individual_features_test/"
FOLDER_CLASS1 = "/work/Sultan/data/feature_vec_1/individual_features_all/"
FOLDER_NEG    = "/work/Sultan/data/feature_vec_tanks/individual_features/"
OUTPUT_FOLDER = "/work/Sultan/results/"

NUM_ITER           = 10
N_SAMPLES_CLASS0   = 9
N_CLUSTERS_KMEANS  = 2
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 3
RANDOM_STATE       = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_all_vectors(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in '{folder}'")

    vectors, names = [], []
    for fname in files:
        vec = np.load(os.path.join(folder, fname))
        vectors.append(vec.flatten())
        names.append(os.path.splitext(fname)[0])

    return np.array(vectors), names


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
def preprocess(X: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ─────────────────────────────────────────────
# KMEANS
# ─────────────────────────────────────────────
def run_kmeans(X_scaled):
    km = KMeans(n_clusters=N_CLUSTERS_KMEANS,
                random_state=RANDOM_STATE,
                n_init=10)

    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_

    distances = np.array([
        np.linalg.norm(X_scaled[i] - centroids[labels[i]])
        for i in range(len(X_scaled))
    ])

    return labels, distances


# ─────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────
def run_dbscan(X_scaled):
    db = DBSCAN(eps=DBSCAN_EPS,
                min_samples=DBSCAN_MIN_SAMPLES)

    labels = db.fit_predict(X_scaled)
    core_labels = [l for l in set(labels) if l != -1]

    distances = np.full(len(X_scaled), np.nan)

    if core_labels:
        centroids = {
            l: X_scaled[labels == l].mean(axis=0)
            for l in core_labels
        }

        for i, label in enumerate(labels):
            if label == -1:
                distances[i] = min(
                    np.linalg.norm(X_scaled[i] - c)
                    for c in centroids.values()
                )
            else:
                distances[i] = np.linalg.norm(
                    X_scaled[i] - centroids[label]
                )
    else:
        pseudo_centroid = X_scaled.mean(axis=0)
        distances = np.array([
            np.linalg.norm(X_scaled[i] - pseudo_centroid)
            for i in range(len(X_scaled))
        ])

    return labels, distances


def label_name(label: int):
    if label == -1:
        return "noise"
    return f"cluster_{label}"


# ─────────────────────────────────────────────
# SAVE TABLE (Dark Styled Like Screenshot)
# ─────────────────────────────────────────────
def save_summary_table(km_c0, db_c0,
                       km_c1, db_c1,
                       km_test, db_test,
                       km_test_labels, db_test_labels,
                       km_neg, db_neg,
                       km_neg_labels, db_neg_labels):

    km_c0_avg = np.mean(km_c0, axis=0)
    db_c0_avg = np.nanmean(db_c0, axis=0)

    rows = []

    # Class 0 rows
    for i in range(N_SAMPLES_CLASS0):
        rows.append([
            f"class0_slot_{i:02d}", "0",
            f"{km_c0_avg[i]:.4f}", "—",
            f"{db_c0_avg[i]:.4f}", "—"
        ])

    # Known class 1
    rows.append([
        "known_class1", "Pickup trucks",
        f"{np.mean(km_c1):.4f}", "—",
        f"{np.nanmean(db_c1):.4f}", "—"
    ])

    # Test vec
    km_test_cluster = Counter(km_test_labels).most_common(1)[0]
    db_test_cluster = Counter(db_test_labels).most_common(1)[0]

    rows.append([
        "test_vec", "Vehicles",
        f"{np.mean(km_test):.4f}",
        f"{label_name(km_test_cluster[0])} ({km_test_cluster[1]}/{NUM_ITER} iters)",
        f"{np.nanmean(db_test):.4f}",
        f"{label_name(db_test_cluster[0])} ({db_test_cluster[1]}/{NUM_ITER} iters)"
    ])

    # Negative control
    km_neg_cluster = Counter(km_neg_labels).most_common(1)[0]
    db_neg_cluster = Counter(db_neg_labels).most_common(1)[0]

    rows.append([
        "negative_control", "Tanks",
        f"{np.mean(km_neg):.4f}",
        f"{label_name(km_neg_cluster[0])} ({km_neg_cluster[1]}/{NUM_ITER} iters)",
        f"{np.nanmean(db_neg):.4f}",
        f"{label_name(db_neg_cluster[0])} ({db_neg_cluster[1]}/{NUM_ITER} iters)"
    ])

    col_labels = [
        "Vector", "Class",
        "KM avg dist", "KM assigned to",
        "DB avg dist", "DB assigned to"
    ]

    # Save CSV
    csv_path = os.path.join(OUTPUT_FOLDER, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        writer.writerows(rows)

    # ── DARK THEMED PNG ──
    fig_h = max(4, 0.55 * (len(rows) + 3))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    ax.axis("off")

    tbl = ax.table(cellText=rows,
                   colLabels=col_labels,
                   loc="center",
                   cellLoc="center")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.9)

    # Header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2A3563")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row styling
    for i, row in enumerate(rows, start=1):
        name = row[0]

        for j in range(len(col_labels)):

            if name == "known_class1":
                tbl[i, j].set_facecolor("#3A1F1F")
                tbl[i, j].set_text_props(color="#FF9999", fontweight="bold")

            elif name == "test_vec":
                tbl[i, j].set_facecolor("#1F3A1F")
                tbl[i, j].set_text_props(color="#88FF88", fontweight="bold")

            elif name == "negative_control":
                tbl[i, j].set_facecolor("#2B1F3A")
                tbl[i, j].set_text_props(color="#C88CFF", fontweight="bold")

            else:
                bg = "#141620" if i % 2 == 0 else "#1A1D2E"
                tbl[i, j].set_facecolor(bg)
                tbl[i, j].set_text_props(color="#DDDDDD")

            tbl[i, j].set_edgecolor("#2F3555")

    ax.set_title(
        f"Average distance to cluster centroid  ({NUM_ITER} iterations)\n"
        "Class-0 slots (white)  |  Known class-1 (red)  |  "
        "test_vec (green)  |  negative_control_tanks (purple)",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    png_path = os.path.join(
        OUTPUT_FOLDER,
        "summary_table_3class_Neg_cont_Tanks_"+str(N_CLUSTERS_KMEANS)+"_centroid.png"
    )

    plt.savefig(
        png_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print("Saved CSV →", csv_path)
    print("Saved PNG →", png_path)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    all_vecs_c0, _ = load_all_vectors(FOLDER_CLASS0)
    all_vecs_c1, _ = load_all_vectors(FOLDER_CLASS1)
    all_vecs_neg, _ = load_all_vectors(FOLDER_NEG)

    km_c0 = np.zeros((NUM_ITER, N_SAMPLES_CLASS0))
    db_c0 = np.zeros((NUM_ITER, N_SAMPLES_CLASS0))

    km_c1, db_c1 = [], []
    km_test, db_test = [], []
    km_test_labels, db_test_labels = [], []

    km_neg, db_neg = [], []
    km_neg_labels, db_neg_labels = [], []

    for it in range(NUM_ITER):

        c0_idx = rng.choice(len(all_vecs_c0),
                            size=N_SAMPLES_CLASS0,
                            replace=False)

        c1_pair = rng.choice(len(all_vecs_c1),
                             size=2,
                             replace=False)

        neg_idx = rng.choice(len(all_vecs_neg),
                             size=1)

        X = np.vstack([
            all_vecs_c0[c0_idx],
            all_vecs_c1[[c1_pair[0]]],
            all_vecs_c1[[c1_pair[1]]],
            all_vecs_neg[[neg_idx[0]]]
        ])

        X_scaled = preprocess(X)

        km_labels, km_dists = run_kmeans(X_scaled)
        db_labels, db_dists = run_dbscan(X_scaled)

        km_c0[it] = km_dists[:9]
        db_c0[it] = db_dists[:9]

        km_c1.append(km_dists[9])
        db_c1.append(db_dists[9])

        km_test.append(km_dists[10])
        db_test.append(db_dists[10])
        km_test_labels.append(km_labels[10])
        db_test_labels.append(db_labels[10])

        km_neg.append(km_dists[11])
        db_neg.append(db_dists[11])
        km_neg_labels.append(km_labels[11])
        db_neg_labels.append(db_labels[11])

    save_summary_table(km_c0, db_c0,
                       km_c1, db_c1,
                       km_test, db_test,
                       km_test_labels, db_test_labels,
                       km_neg, db_neg,
                       km_neg_labels, db_neg_labels)

    print("\n✅ Done (styled output generated).")