"""
Clustering Analysis: KMeans & DBSCAN
Per iteration (11 vectors total):
  - 9 randomly sampled from class-0
  - 1 randomly sampled from class-1  (the known outlier)
  - 1 randomly sampled from class-1  (test_vec, different from the first)

Final table:
  - Average distance per slot for class-0 (slots 0..8)
  - Average distance of known class-1 vector across iterations
  - test_vec: avg distance to its assigned centroid + most frequent cluster assignment
Output: summary_table.csv + summary_table.png
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
FOLDER_CLASS0      = "/home/user1/ariel/Sultan/data/feature_vec_0/individual_features_test"
FOLDER_CLASS1      = "/home/user1/ariel/Sultan/data/feature_vec_1/individual_features_all"
OUTPUT_FOLDER      = "/home/user1/ariel/Sultan/data/results"

NUM_ITER           = 20
N_SAMPLES_CLASS0   = 9
N_CLUSTERS_KMEANS  = 2      # single centroid — distance reflects true separation
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 3
RANDOM_STATE       = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)

# Fixed positions in the stacked array each iteration:
# indices 0..8  → class-0 slots
# index   9     → known class-1 vector
# index   10    → test_vec


# ─────────────────────────────────────────────
# 1. LOAD
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
# 2. PREPROCESS (scaling only — cluster on full dims)
# ─────────────────────────────────────────────
def preprocess(X: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ─────────────────────────────────────────────
# 3. KMEANS
# ─────────────────────────────────────────────
def run_kmeans(X_scaled):
    km = KMeans(n_clusters=N_CLUSTERS_KMEANS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    distances = np.array([
        np.linalg.norm(X_scaled[i] - centroids[labels[i]])
        for i in range(len(X_scaled))
    ])
    return labels, distances


# ─────────────────────────────────────────────
# 4. DBSCAN
# ─────────────────────────────────────────────
def run_dbscan(X_scaled):
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(X_scaled)
    core_labels = [l for l in set(labels) if l != -1]
    distances = np.full(len(X_scaled), np.nan)

    if core_labels:
        centroids = {l: X_scaled[labels == l].mean(axis=0) for l in core_labels}
        for i, label in enumerate(labels):
            if label == -1:
                distances[i] = min(
                    np.linalg.norm(X_scaled[i] - c) for c in centroids.values()
                )
            else:
                distances[i] = np.linalg.norm(X_scaled[i] - centroids[label])
    else:
        pseudo_centroid = X_scaled.mean(axis=0)
        distances = np.array([
            np.linalg.norm(X_scaled[i] - pseudo_centroid)
            for i in range(len(X_scaled))
        ])

    return labels, distances


# ─────────────────────────────────────────────
# 5. CLUSTER LABEL → human-readable name
# ─────────────────────────────────────────────
def label_name(label: int) -> str:
    if label == -1:
        return "noise"
    return f"cluster_{label}"


# ─────────────────────────────────────────────
# 6. SAVE TABLE — csv + png
# ─────────────────────────────────────────────
def save_summary_table(km_c0, db_c0,
                       km_c1_list, db_c1_list,
                       km_test_dists, db_test_dists,
                       km_test_labels, db_test_labels):
    """
    km_c0 / db_c0        : (NUM_ITER, N_SAMPLES_CLASS0) distances per slot
    km_c1_list           : list[float]  known class-1 distance per iter
    db_c1_list           : list[float]
    km_test_dists        : list[float]  test_vec distance per iter
    db_test_dists        : list[float]
    km_test_labels       : list[int]    cluster assignment of test_vec per iter
    db_test_labels       : list[int]
    """

    # Averages
    km_c0_avg = np.mean(km_c0, axis=0)
    db_c0_avg = np.nanmean(db_c0, axis=0)
    km_c1_avg = np.mean(km_c1_list)
    db_c1_avg = np.nanmean(db_c1_list)
    km_test_avg = np.mean(km_test_dists)
    db_test_avg = np.nanmean(db_test_dists)

    # Most frequent cluster assignment for test_vec
    km_test_most_common = Counter(km_test_labels).most_common(1)[0]
    db_test_most_common = Counter(db_test_labels).most_common(1)[0]
    km_test_cluster = (f"{label_name(km_test_most_common[0])} "
                       f"({km_test_most_common[1]}/{NUM_ITER} iters)")
    db_test_cluster = (f"{label_name(db_test_most_common[0])} "
                       f"({db_test_most_common[1]}/{NUM_ITER} iters)")

    # Build rows
    col_labels = ["Vector", "Class", "KM avg dist", "KM assigned to",
                  "DB avg dist", "DB assigned to"]
    rows = []

    for i in range(N_SAMPLES_CLASS0):
        rows.append([f"class0_slot_{i:02d}", "0",
                     f"{km_c0_avg[i]:.4f}", "—",
                     f"{db_c0_avg[i]:.4f}", "—"])

    rows.append(["known_class1", "1",
                 f"{km_c1_avg:.4f}", "—",
                 f"{db_c1_avg:.4f}", "—"])

    rows.append(["test_vec", "1",
                 f"{km_test_avg:.4f}", km_test_cluster,
                 f"{db_test_avg:.4f}", db_test_cluster])

    # ── CSV ──
    csv_path = os.path.join(OUTPUT_FOLDER, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        writer.writerows(rows)
    print(f"📄 CSV  saved → {csv_path}")

    # ── console ──
    print("\n" + "═" * 95)
    print("  SUMMARY: Average distance per vector slot")
    print("═" * 95)
    print(f"  {'Vector':<22} {'Class':>5} {'KM avg':>10} {'KM assigned to':<28} "
          f"{'DB avg':>10} {'DB assigned to'}")
    print("  " + "-" * 93)
    for row in rows:
        print(f"  {row[0]:<22} {row[1]:>5} {row[2]:>10} {row[3]:<28} {row[4]:>10} {row[5]}")
    print("═" * 95)

    # ── PNG ──
    n_rows = len(rows)
    fig_h = max(3, 0.5 * (n_rows + 3))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1F2B5E")
        tbl[0, j].set_text_props(color="#FFFFFF", fontweight="bold")

    # Row colours
    for i, row in enumerate(rows, start=1):
        cls = row[1]
        is_test = (row[0] == "test_vec")
        for j in range(len(col_labels)):
            if is_test:
                tbl[i, j].set_facecolor("#1F3A1F")
                tbl[i, j].set_text_props(color="#88FF88", fontweight="bold")
            elif cls == "1":
                tbl[i, j].set_facecolor("#3A1F1F")
                tbl[i, j].set_text_props(color="#FF9999", fontweight="bold")
            else:
                bg = "#1A1D2E" if i % 2 == 0 else "#141620"
                tbl[i, j].set_facecolor(bg)
                tbl[i, j].set_text_props(color="#DDDDDD")
            tbl[i, j].set_edgecolor("#333355")

    ax.set_title(
        f"Average distance to cluster centroid  ({NUM_ITER} iterations)\n"
        "Class-0 slots (white)  |  Known class-1 (red)  |  test_vec (green)",
        color="#FFFFFF", fontsize=10, fontweight="bold", pad=14
    )

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_FOLDER, "summary_table_2clsKM_3samDBSCAN.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"🖼  PNG  saved → {png_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print(f"Loading class-0 vectors from: {FOLDER_CLASS0}")
    all_vecs_c0, all_names_c0 = load_all_vectors(FOLDER_CLASS0)
    print(f"  → {len(all_names_c0)} vectors found")

    print(f"Loading class-1 vectors from: {FOLDER_CLASS1}")
    all_vecs_c1, all_names_c1 = load_all_vectors(FOLDER_CLASS1)
    print(f"  → {len(all_names_c1)} vectors found")

    if len(all_vecs_c1) < 2:
        raise ValueError("Need at least 2 class-1 vectors: one for known outlier, one for test_vec.")

    # Accumulators
    km_c0        = np.zeros((NUM_ITER, N_SAMPLES_CLASS0))
    db_c0        = np.zeros((NUM_ITER, N_SAMPLES_CLASS0))
    km_c1_list   = []
    db_c1_list   = []
    km_test_dists  = []
    db_test_dists  = []
    km_test_labels = []
    db_test_labels = []

    for it in range(NUM_ITER):
        print(f"\n  ITERATION {it+1:02d} / {NUM_ITER}")

        # Sample 9 from class-0
        c0_indices = rng.choice(len(all_vecs_c0), size=N_SAMPLES_CLASS0, replace=False)

        # Sample 2 DIFFERENT vectors from class-1
        c1_pair = rng.choice(len(all_vecs_c1), size=2, replace=False)
        c1_known_idx = c1_pair[0]
        c1_test_idx  = c1_pair[1]

        c1_known_name = all_names_c1[c1_known_idx]
        c1_test_name  = all_names_c1[c1_test_idx]

        print(f"    Class-0 sampled  : {[all_names_c0[i] for i in c0_indices]}")
        print(f"    Class-1 known    : {c1_known_name}")
        print(f"    Class-1 test_vec : {c1_test_name}")

        # Stack: positions 0..8 = class-0, 9 = known c1, 10 = test_vec
        X = np.vstack([
            all_vecs_c0[c0_indices],
            all_vecs_c1[[c1_known_idx]],
            all_vecs_c1[[c1_test_idx]]
        ])
        X_scaled = preprocess(X)

        km_labels, km_dists = run_kmeans(X_scaled)
        db_labels, db_dists = run_dbscan(X_scaled)

        # Store class-0 slot distances
        km_c0[it, :] = km_dists[:N_SAMPLES_CLASS0]
        db_c0[it, :] = db_dists[:N_SAMPLES_CLASS0]

        # Store known class-1 distances
        km_c1_list.append(km_dists[N_SAMPLES_CLASS0])
        db_c1_list.append(db_dists[N_SAMPLES_CLASS0])

        # Store test_vec distances and cluster assignments
        km_test_dists.append(km_dists[N_SAMPLES_CLASS0 + 1])
        db_test_dists.append(db_dists[N_SAMPLES_CLASS0 + 1])
        km_test_labels.append(km_labels[N_SAMPLES_CLASS0 + 1])
        db_test_labels.append(db_labels[N_SAMPLES_CLASS0 + 1])

        print(f"    KM  dists → c0: {np.round(km_dists[:N_SAMPLES_CLASS0], 3)}  "
              f"known_c1: {km_dists[N_SAMPLES_CLASS0]:.3f}  "
              f"test_vec: {km_dists[N_SAMPLES_CLASS0+1]:.3f} (cluster {km_labels[N_SAMPLES_CLASS0+1]})")
        print(f"    DB  dists → c0: {np.round(db_dists[:N_SAMPLES_CLASS0], 3)}  "
              f"known_c1: {db_dists[N_SAMPLES_CLASS0]:.3f}  "
              f"test_vec: {db_dists[N_SAMPLES_CLASS0+1]:.3f} (cluster {db_labels[N_SAMPLES_CLASS0+1]})")

    save_summary_table(km_c0, db_c0,
                       km_c1_list, db_c1_list,
                       km_test_dists, db_test_dists,
                       km_test_labels, db_test_labels)
    print("\n✅ Done.")
