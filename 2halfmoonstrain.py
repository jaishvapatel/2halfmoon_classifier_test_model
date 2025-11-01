
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  

# Load dataset
data = pd.read_csv("2halfmoonsTrain.csv")
X = data[['X', 'Y']].values
y = (data['ClassLabel'] - 1).values  


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(10, 5),
                    activation='tanh',
                    solver='adam',
                    learning_rate_init=0.01,
                    max_iter=1000,
                    random_state=42)

# Train model (on scaled data)
mlp.fit(X_train_s, y_train)
# === Save trained sklearn model + scaler (after mlp.fit) ===
import joblib, os
model_out = "trained_two_moons_model_from_fit.joblib"
scaler_out = "trained_two_moons_scaler_from_fit.joblib"

joblib.dump(mlp, model_out)
joblib.dump(scaler, scaler_out)
print(f"Saved model to: {model_out}")
print(f"Saved scaler to: {scaler_out}")

# Optional: quick evaluation on your validation/test split and save a figure
try:
    import numpy as np
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt

    # If X_test_s and y_test exist in this scope use them, otherwise build from earlier splits
    Xt = globals().get('X_test_s', None)
    yt = globals().get('y_test', None)
    if Xt is None or yt is None:
        Xt = X_test_s  
        yt = y_test

    y_pred = mlp.predict(Xt)
    cm = confusion_matrix(yt, y_pred)
    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]
    acc = accuracy_score(yt, y_pred)
    prec = precision_score(yt, y_pred)
    rec = recall_score(yt, y_pred)
    f1 = f1_score(yt, y_pred)

    print("Confusion (counts):\n", cm)
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # plot predictions (scaled back to original space for plotting)
    # If Xt is scaled, we need to inverse_transform
    try:
        orig_grid = scaler.inverse_transform(Xt)
    except Exception:
        orig_grid = Xt

    plt.figure(figsize=(5,4))
    plt.scatter(orig_grid[:,0], orig_grid[:,1], c=y_pred, cmap='bwr', s=20, edgecolor='k', alpha=0.8)
    plt.title('Test set predictions (saved model from fit)')
    plt.xlabel('X'); plt.ylabel('Y')
    plt.tight_layout()
    figname = "test_predictions_from_fit.png"
    plt.savefig(figname, dpi=200)
    plt.close()
    print("Saved figure:", figname)
except Exception as e:
    print("Quick-eval skipped or failed:", e)


# Calculate accuracies (on scaled data)
train_acc = accuracy_score(y_train, mlp.predict(X_train_s))
val_acc   = accuracy_score(y_val,   mlp.predict(X_val_s))
test_acc  = accuracy_score(y_test,  mlp.predict(X_test_s))

print(f"Training Accuracy:  {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy:      {test_acc:.4f}")

# --- 5-Fold Cross Validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(mlp, X, y, cv=kf, scoring='accuracy')
print(f"\nCross-Validation Accuracies: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# --- Decision Boundary Plot---
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-0.2, X[:,0].max()+0.2, 300),
    np.linspace(X[:,1].min()-0.2, X[:,1].max()+0.2, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]
grid_s = scaler.transform(grid)                                # scale the grid
proba = mlp.predict_proba(grid_s)[:, 1].reshape(xx.shape)      # p(class=1)
Z = (proba >= 0.5).astype(int)                                 # for background coloring

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, proba, levels=20, alpha=0.4, cmap=plt.cm.RdYlGn)  # soft background (optional)
plt.contour(xx, yy, proba, levels=[0.5], colors='k', linewidths=2)     # <-- required decision boundary
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlGn, edgecolor='k')
plt.gca().set_aspect("equal", adjustable="box")                         # keep moons round
plt.title("Decision Boundary for Two-Moons Classification")
plt.xlabel("X"); plt.ylabel("Y")

plt.show()

# --- Learning Curve Plot (unchanged) ---
plt.figure(figsize=(8,5))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.title("Learning Curve - Loss")
plt.xlabel("Iterations"); plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nColumns in dataset:", data.columns.tolist())


# ==========================================================
# Two Half Moons Classification – Accuracy Curve (Partial Fit)
# ==========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1) Load dataset ---
df = pd.read_csv("2halfmoonsTrain.csv")  # Ensure this CSV is in the same folder
X = df[["X", "Y"]].values
y = (df["ClassLabel"].values - 1).astype(int)  # Convert {1,2} → {0,1}

# --- 2) Split into training & validation sets ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3) Standardize inputs (train only) ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# --- 4) Define the MLP model ---
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation="tanh",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1,          
    warm_start=True,    
    random_state=42
)

# --- 5) Train using partial_fit and record accuracy ---
classes = np.array([0, 1])
epochs = 300
batch_size = 64
train_acc, val_acc = [], []

# Initialize model once
mlp.partial_fit(X_train_s[:batch_size], y_train[:batch_size], classes=classes)

for _ in range(epochs):
    # Shuffle batches each epoch
    idx = np.random.permutation(len(X_train_s))
    for start in range(0, len(X_train_s), batch_size):
        end = start + batch_size
        mlp.partial_fit(X_train_s[idx[start:end]], y_train[idx[start:end]])

    # Record accuracy (%) for this epoch
    train_acc.append(accuracy_score(y_train, mlp.predict(X_train_s)) * 100)
    val_acc.append(accuracy_score(y_val, mlp.predict(X_val_s)) * 100)

# --- 6) Plot Accuracy Curve ---
plt.figure(figsize=(6, 4))
plt.plot(train_acc, label="Training Accuracy", linewidth=2)
plt.plot(val_acc, label="Validation Accuracy", linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curve – Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve_partialfit.png", dpi=300, bbox_inches="tight")
plt.show()

print("Accuracy curve saved as 'accuracy_curve_partialfit.png'")