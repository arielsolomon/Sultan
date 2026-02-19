"""
Clustering Analysis: KMeans & DBSCAN
- Each iteration: 9 randomly sampled class-0 vectors + 1 randomly sampled class-1 vector
- Distances are tracked by POSITION (slot 0..8 for class-0, slot 0 for class-1)
- Final table: average distance per position across all iterations
- Output: summary_table.csv + summary_table.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FOLDER_CLASS0      = "/home/user1/ariel/Sultan/data/feature_vec_0/individual_features_test"
FOLDER_CLASS1      = "/home/user1/ariel/Sultan/data/feature_vec_1/individual_features_all"
OUTPUT_FOLDER      = "/home/user1/ariel/Sultan/data/results"

NUM_ITER           = 20     # ← number of iterations
N_SAMPLES_CLASS0   = 9      # class-0 vectors sampled per iteration
N_CLUSTERS_KMEANS  = 2
DBSCAN_EPS         = 0.5
DBSCAN_MIN_SAMPLES = 3
RANDOM_STATE       = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


# ─────────────────────────────────────────────
# 1. LOAD ALL VECTORS FROM A FOLDER
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
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA only if needed for dimensionality reduction before clustering
    # clustering is done on full scaled space
    return X_scaled


# ─────────────────────────────────────────────
# 3. KMEANS — distance of each vector to its assigned centroid
# ─────────────────────────────────────────────
def run_kmeans(X_scaled):
    km = KMeans(n_clusters=N_CLUSTERS_KMEANS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    distances = np.array([
        np.linalg.norm(X_scaled[i] - centroids[labels[i]])
        for i in range(len(X_scaled))
    ])
    return distances


# ─────────────────────────────────────────────
# 4. DBSCAN — distance to own centroid or nearest centroid if noise
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
        # All noise — use global centroid
        pseudo_centroid = X_scaled.mean(axis=0)
        for i in range(len(X_scaled)):
            distances[i] = np.linalg.norm(X_scaled[i] - pseudo_centroid)

    return distances


# ─────────────────────────────────────────────
# 5. SAVE SUMMARY TABLE — csv + image
# ─────────────────────────────────────────────
def save_summary_table(km_c0, db_c0, km_c1_list, db_c1_list):
    """
    km_c0 / db_c0 : shape (NUM_ITER, N_SAMPLES_CLASS0) — distances per slot per iter
    km_c1_list     : list of NUM_ITER scalar distances for the class-1 vector
    db_c1_list     : same for DBSCAN
    """

    # Average across iterations for each position slot
    km_c0_avg = np.mean(km_c0, axis=0)   # shape (N_SAMPLES_CLASS0,)
    db_c0_avg = np.nanmean(db_c0, axis=0)

    km_c1_avg = np.mean(km_c1_list)
    db_c1_avg = np.nanmean(db_c1_list)

    # Build rows: one per class-0 slot + one for class-1
    col_labels = ["Vector slot", "Class", "KM avg dist", "DB avg dist"]
    rows = []
    for i in range(N_SAMPLES_CLASS0):
        rows.append([f"class0_slot_{i:02d}", "0",
                     f"{km_c0_avg[i]:.4f}", f"{db_c0_avg[i]:.4f}"])
    rows.append(["class1 (avg over iters)", "1",
                 f"{km_c1_avg:.4f}", f"{db_c1_avg:.4f}"])

    # ── CSV ──
    csv_path = os.path.join(OUTPUT_FOLDER, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        writer.writerows(rows)
    print(f"📄 CSV saved  → {csv_path}")

    # ── console ──
    print("\n" + "═" * 65)
    print("  SUMMARY: Average distance per vector slot")
    print("═" * 65)
    print(f"  {'Vector slot':<28} {'Class':>5} {'KM avg':>10} {'DB avg':>10}")
    print("  " + "-" * 57)
    for row in rows:
        print(f"  {row[0]:<28} {row[1]:>5} {row[2]:>10} {row[3]:>10}")
    print("═" * 65)

    # ── PNG table ──
    n_rows = len(rows)
    fig_h = max(3, 0.45 * (n_rows + 3))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # Header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1F2B5E")
        tbl[0, j].set_text_props(color="#FFFFFF", fontweight="bold")

    # Row styles — class-0 white, class-1 red
    for i, row in enumerate(rows, start=1):
        is_c1 = (row[1] == "1")
        for j in range(len(col_labels)):
            if is_c1:
                tbl[i, j].set_facecolor("#3A1F1F")
                tbl[i, j].set_text_props(color="#FF9999", fontweight="bold")
            else:
                bg = "#1A1D2E" if i % 2 == 0 else "#141620"
                tbl[i, j].set_facecolor(bg)
                tbl[i, j].set_text_props(color="#DDDDDD")
            tbl[i, j].set_edgecolor("#333355")

    ax.set_title(
        f"Average distance to cluster centroid per vector slot  ({NUM_ITER} iterations)\n"
        "Class-0 slots = same index across runs  |  Class-1 = single vector averaged over all runs",
        color="#FFFFFF", fontsize=10, fontweight="bold", pad=14
    )

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_FOLDER, "summary_table.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"🖼  PNG saved  → {png_path}")


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

    # Storage: rows = iterations, cols = positions
    km_c0 = np.zeros((NUM_ITER, N_SAMPLES_CLASS0))
    db_c0 = np.zeros((NUM_ITER, N_SAMPLES_CLASS0))
    km_c1_list = []
    db_c1_list = []

    for it in range(NUM_ITER):
        print(f"\n  ITERATION {it+1:02d} / {NUM_ITER}")

        # Random sample: 9 from class-0, 1 from class-1
        c0_indices = rng.choice(len(all_vecs_c0), size=N_SAMPLES_CLASS0, replace=False)
        c1_index   = rng.choice(len(all_vecs_c1))

        c0_names = [all_names_c0[i] for i in c0_indices]
        c1_name  = all_names_c1[c1_index]
        print(f"    Class-0 sampled : {c0_names}")
        print(f"    Class-1 sampled : {c1_name}")

        # Stack: first 9 = class-0 (positions 0..8), last 1 = class-1 (position 9)
        X = np.vstack([all_vecs_c0[c0_indices], all_vecs_c1[[c1_index]]])
        X_scaled = preprocess(X)

        km_dists = run_kmeans(X_scaled)
        db_dists = run_dbscan(X_scaled)

        # Store by position
        km_c0[it, :] = km_dists[:N_SAMPLES_CLASS0]
        db_c0[it, :] = db_dists[:N_SAMPLES_CLASS0]
        km_c1_list.append(km_dists[N_SAMPLES_CLASS0])
        db_c1_list.append(db_dists[N_SAMPLES_CLASS0])

        print(f"    KM  dists (c0): {np.round(km_dists[:N_SAMPLES_CLASS0], 3)}  c1: {km_dists[N_SAMPLES_CLASS0]:.3f}")
        print(f"    DB  dists (c0): {np.round(db_dists[:N_SAMPLES_CLASS0], 3)}  c1: {db_dists[N_SAMPLES_CLASS0]:.3f}")

    save_summary_table(km_c0, db_c0, km_c1_list, db_c1_list)
    print("\n✅ Done.")