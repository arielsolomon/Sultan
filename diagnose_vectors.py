#!/usr/bin/env python3
"""
Diagnostic script — run this BEFORE the pipeline to verify that your
.npy feature vectors are actually separable.

It checks:
  1. Raw cosine similarity within vs across classes
  2. PCA 2D plot  (saved to disk)
  3. t-SNE 2D plot (saved to disk)
  4. KMeans purity with increasing n_samples
"""
import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────────
VEC0_PATH  = '/work/Sultan/data/rotem_non_pickups_f_vectors/individual_features/'
VEC1_PATH  = '/work/Sultan/data/rotem_pickups_f_vectors/individual_features/'
SAVE_DIR   = '/work/Sultan/runs/diagnostics/'
N_SAMPLES  = 24   # must match N_TRAIN_SAMPLES in main pipeline
# ──────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Load vectors ────────────────────────────────────────────────────────
def load_all(path):
    files = glob.glob(os.path.join(path, '*.npy'))
    return np.vstack([np.load(f) for f in files])

print("[1/5] Loading ALL available vectors (not just 24) …")
X0_all = load_all(VEC0_PATH)   # non-pickup
X1_all = load_all(VEC1_PATH)   # pickup
print(f"      non-pickup : {X0_all.shape}")
print(f"      pickup     : {X1_all.shape}")

# Use min available so classes are balanced
n = min(len(X0_all), len(X1_all), 50)
X0 = X0_all[:n]
X1 = X1_all[:n]

X    = np.vstack([X0, X1])
y_gt = np.array([0]*n + [1]*n)   # ground-truth labels

# ── 2. Cosine similarity ───────────────────────────────────────────────────
print("\n[2/5] Cosine similarity …")
X_norm = normalize(X)
sim_matrix = cosine_similarity(X_norm)

within0  = sim_matrix[:n, :n]
within1  = sim_matrix[n:, n:]
across   = sim_matrix[:n, n:]

# Exclude diagonal (self-similarity = 1.0)
np.fill_diagonal(within0, np.nan)
np.fill_diagonal(within1, np.nan)

print(f"  within non-pickup  : {np.nanmean(within0):.4f}  (higher = more similar)")
print(f"  within pickup      : {np.nanmean(within1):.4f}")
print(f"  across classes     : {np.nanmean(across):.4f}  (lower = more separable)")
print(f"  separation ratio   : {np.nanmean(across) / max(np.nanmean(within0), np.nanmean(within1)):.4f}  (lower = better)")

# ── 3. PCA plot ────────────────────────────────────────────────────────────
print("\n[3/5] PCA plot …")
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca  = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:n, 0], X_2d[:n, 1], c='steelblue', label='non_pickup', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.scatter(X_2d[n:, 0], X_2d[n:, 1], c='tomato',    label='pickup',     alpha=0.7, edgecolors='k', linewidths=0.3)
plt.title(f'PCA — explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%')
plt.legend()
plt.tight_layout()
pca_path = os.path.join(SAVE_DIR, 'pca_vectors.png')
plt.savefig(pca_path, dpi=150)
plt.close()
print(f"  Saved → {pca_path}")

# ── 4. t-SNE plot ──────────────────────────────────────────────────────────
print("\n[4/5] t-SNE plot (may take ~30s) …")
tsne = TSNE(n_components=2, perplexity=min(15, n-1), random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:n, 0], X_tsne[:n, 1], c='steelblue', label='non_pickup', alpha=0.7, edgecolors='k', linewidths=0.3)
plt.scatter(X_tsne[n:, 0], X_tsne[n:, 1], c='tomato',    label='pickup',     alpha=0.7, edgecolors='k', linewidths=0.3)
plt.title('t-SNE — feature vector distribution')
plt.legend()
plt.tight_layout()
tsne_path = os.path.join(SAVE_DIR, 'tsne_vectors.png')
plt.savefig(tsne_path, dpi=150)
plt.close()
print(f"  Saved → {tsne_path}")

# ── 5. KMeans purity sweep ────────────────────────────────────────────────
print("\n[5/5] KMeans purity sweep …")
def kmeans_purity(X_sc, labels_gt):
    km = KMeans(n_clusters=2, random_state=42, n_init=24)
    km.fit(X_sc)
    pred = km.labels_
    # Try both label assignments, return best
    acc0 = np.mean(pred == labels_gt)
    acc1 = np.mean(pred != labels_gt)   # flipped
    return max(acc0, acc1), km.labels_

purity, km_labels = kmeans_purity(X_scaled, y_gt)
print(f"  KMeans purity (best label assignment) : {purity*100:.1f}%")
print(f"  Cluster sizes : 0→{(km_labels==0).sum()}  1→{(km_labels==1).sum()}")

# Per-class breakdown
for cls, name in [(0, 'non_pickup'), (1, 'pickup')]:
    gt_mask  = y_gt == cls
    dominant = np.bincount(km_labels[gt_mask]).argmax()
    correct  = (km_labels[gt_mask] == dominant).mean()
    print(f"  {name:12s} → mostly cluster {dominant}  ({correct*100:.0f}% consistent)")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n── SUMMARY ──────────────────────────────────────────────────────────")
sep = np.nanmean(across) / max(np.nanmean(within0), np.nanmean(within1))
if sep < 0.92:
    print("  ✓ Cosine similarity suggests reasonable class separation.")
else:
    print("  ✗ Cosine similarity shows classes are very similar — poor separability.")

if purity >= 0.75:
    print(f"  ✓ KMeans purity {purity*100:.1f}% — clusters are meaningful.")
elif purity >= 0.60:
    print(f"  ~ KMeans purity {purity*100:.1f}% — weak but present separation.")
else:
    print(f"  ✗ KMeans purity {purity*100:.1f}% — no meaningful clustering found.")
    print("    → Consider switching to a supervised classifier (SVM/logistic regression).")

print(f"\n  Check the PCA and t-SNE plots in: {SAVE_DIR}")
print("────────────────────────────────────────────────────────────────────")
