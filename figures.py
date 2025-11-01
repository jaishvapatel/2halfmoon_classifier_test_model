# make_figures.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors

# =========== Configuration ===========
MODEL_FILE  = "trained_two_moons_model_from_fit.joblib"
SCALER_FILE = "trained_two_moons_scaler_from_fit.joblib"
TEST_CSV    = "2halfmoonsTest.csv"   # or "2halfmoonsTest (1).csv"

OUT_CM_COUNTS = "confusion_counts.png"
OUT_CM_NORM   = "confusion_normalized.png"
OUT_DECISION  = "decision_boundary.png"
OUT_CONF_SUM  = "prediction_confidence.txt"
OUT_CONF_HIST = "confidence_histogram.png"
# ======================================

# --- Load model, scaler, and test data ---
assert os.path.exists(MODEL_FILE), f"Model file not found: {MODEL_FILE}"
assert os.path.exists(SCALER_FILE), f"Scaler file not found: {SCALER_FILE}"
assert os.path.exists(TEST_CSV), f"Test CSV not found: {TEST_CSV}"

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
df = pd.read_csv(TEST_CSV)

X_test = df[['X','Y']].values
y_test = (df['ClassLabel'].values - 1).astype(int)

X_test_s = scaler.transform(X_test)
y_pred = model.predict(X_test_s)
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test_s)[:,1]
else:
    from scipy.special import expit
    try:
        y_proba = expit(model.decision_function(X_test_s))
    except Exception:
        y_proba = np.zeros(len(X_test_s))

# ========= 1) Confusion matrix (counts) =========
cm = confusion_matrix(y_test, y_pred)

# --- custom teal-purple colormap (unique look) ---
colors_counts = [(0.9, 0.97, 0.95), (0.25, 0.4, 0.45), (0.45, 0.25, 0.55)]
cmap_counts = mcolors.LinearSegmentedColormap.from_list("custom_counts", colors_counts, N=256)

plt.figure(figsize=(3.6,3.6))
plt.imshow(cm, cmap=cmap_counts, interpolation='nearest', vmin=0, vmax=cm.max() if cm.max()>0 else 1)
plt.title("Confusion Matrix (Counts)", fontsize=11)
cbar = plt.colorbar(shrink=0.8)
cbar.ax.tick_params(labelsize=9)
plt.xticks([0,1], ['Pred 0','Pred 1'], fontsize=10)
plt.yticks([0,1], ['True 0','True 1'], fontsize=10)

# overlay numeric text
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i,j]
        txt_color = 'white' if val > cm.max()/2 else 'black'
        plt.text(j, i, str(val), ha='center', va='center', color=txt_color, fontsize=12, fontweight='bold')

plt.ylabel('True label', fontsize=10)
plt.xlabel('Predicted label', fontsize=10)
plt.tight_layout()
plt.savefig(OUT_CM_COUNTS, dpi=300)
plt.close()
print("Saved:", OUT_CM_COUNTS)

# ========= 2) Confusion matrix (normalized %) =========
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
cm_norm_pct = cm_norm * 100.0

plt.figure(figsize=(3.6,3.6))
plt.imshow(cm_norm_pct, cmap='viridis', vmin=0, vmax=100, interpolation='nearest')
plt.title("Confusion Matrix (Normalized %)", fontsize=11)
cbar = plt.colorbar(shrink=0.8)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{int(val)}%"))
cbar.ax.tick_params(labelsize=9)

plt.xticks([0,1], ['Pred 0','Pred 1'], fontsize=10)
plt.yticks([0,1], ['True 0','True 1'], fontsize=10)

for i in range(cm_norm_pct.shape[0]):
    for j in range(cm_norm_pct.shape[1]):
        val = cm_norm_pct[i,j]
        txt_color = 'white' if val > 50 else 'black'
        plt.text(j, i, f"{val:.0f}%", ha='center', va='center', color=txt_color, fontsize=12, fontweight='bold')

plt.ylabel('True label', fontsize=10)
plt.xlabel('Predicted label', fontsize=10)
plt.tight_layout()
plt.savefig(OUT_CM_NORM, dpi=300)
plt.close()
print("Saved:", OUT_CM_NORM)

# ========= 3) Decision boundary + test points =========
pad = 0.4
x_min, x_max = X_test[:,0].min() - pad, X_test[:,0].max() + pad
y_min, y_max = X_test[:,1].min() - pad, X_test[:,1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_s = scaler.transform(grid)

if hasattr(model, "predict_proba"):
    grid_proba = model.predict_proba(grid_s)[:,1].reshape(xx.shape)
else:
    from scipy.special import expit
    try:
        grid_dec = model.decision_function(grid_s).reshape(xx.shape)
        grid_proba = expit(grid_dec)
    except Exception:
        grid_proba = np.zeros_like(xx)

# distinct purple-orange colormap
colors_decision = [(0.97, 0.95, 0.85), (0.85, 0.55, 0.35), (0.4, 0.2, 0.5)]
cmap_decision = mcolors.LinearSegmentedColormap.from_list("custom_boundary", colors_decision, N=256)

plt.figure(figsize=(7,5))
plt.contourf(xx, yy, grid_proba, levels=50, cmap=cmap_decision, alpha=0.8)
CS = plt.contour(xx, yy, grid_proba, levels=[0.5], colors='k', linewidths=2)
CS.collections[0].set_label('Decision boundary (p=0.5)')

correct_mask = (y_pred == y_test)
incorrect_mask = ~correct_mask

plt.scatter(X_test[correct_mask,0], X_test[correct_mask,1],
            c=y_test[correct_mask], cmap='bwr', marker='o', edgecolor='k', s=60, label='Correct')
if incorrect_mask.any():
    plt.scatter(X_test[incorrect_mask,0], X_test[incorrect_mask,1],
                c=y_test[incorrect_mask], cmap='bwr', marker='X', edgecolor='k', s=90, label='Incorrect')

plt.xlabel('X', fontsize=11)
plt.ylabel('Y', fontsize=11)
plt.title('Decision Boundary and Test Predictions', fontsize=12)
plt.legend(loc='upper right', fontsize=9)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tight_layout()
plt.savefig(OUT_DECISION, dpi=300)
plt.close()
print("Saved:", OUT_DECISION)

# ========= 4) Prediction confidence (per predicted label) =========
# compute confidence as probability of the predicted class
# For each sample, if model predicted class 1 -> use P(class=1)
# otherwise use P(class=0) = 1 - P(class=1). So confidence = max(p, 1-p)
conf_per_sample = np.maximum(y_proba, 1.0 - y_proba)

conf_mean = float(np.mean(conf_per_sample))
conf_min  = float(np.min(conf_per_sample))
conf_std  = float(np.std(conf_per_sample))

print(f"Prediction confidence (per predicted label): mean={conf_mean:.4f}, min={conf_min:.4f}, std={conf_std:.4f}")

with open(OUT_CONF_SUM, "w") as f:
    f.write(f"mean {conf_mean:.6f}\nmin {conf_min:.6f}\nstd {conf_std:.6f}\n")
print("Saved:", OUT_CONF_SUM)

# also save a histogram of confidence values for inspection
plt.figure(figsize=(4,2.6))
plt.hist(conf_per_sample, bins=20, range=(0.0,1.0), edgecolor='k', alpha=0.85)
plt.xlabel('Prediction confidence (probability of predicted class)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Confidence histogram', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_CONF_HIST, dpi=200)
plt.close()
print("Saved:", OUT_CONF_HIST)
