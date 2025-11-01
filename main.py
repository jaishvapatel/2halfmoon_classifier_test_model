# evaluate_two_moons.py
"""
Evaluation script for the two-moons test dataset.

Usage examples:
  # Use your saved sklearn model + scaler:
  python evaluate_two_moons.py --test "2halfmoonsTest (1).csv" --model trained_two_moons_model_from_fit.joblib --scaler trained_two_moons_scaler_from_fit.joblib

  # If you only have a model (no scaler saved) and it expects raw inputs:
  python evaluate_two_moons.py --test "2halfmoonsTest (1).csv" --model my_model.joblib

  # If no model provided, the script will train a default model (for convenience)
  python evaluate_two_moons.py --test "2halfmoonsTest (1).csv"
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --------- Default configuration ----------
TEST_CSV = "2halfmoonsTest.csv"
OUT_MODEL = "two_moons_mlp_model.joblib"
OUT_FIG = "test_predictions_scatter.png"
# -----------------------------------------

def train_default_model(random_state=42):
    X_train, y_train = make_moons(n_samples=2000, noise=0.2, random_state=random_state)
    clf = MLPClassifier(hidden_layer_sizes=(16,16), activation='relu', max_iter=2000, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf, None  # default model trained on unscaled data

def load_test_csv(path):
    df = pd.read_csv(path)
    if {'X','Y','ClassLabel'}.issubset(df.columns):
        X = df[['X','Y']].values
        y = (df['ClassLabel'].values - 1).astype(int)  # convert 1/2 -> 0/1
        return X, y
    else:
        raise ValueError("Test CSV must contain columns: X, Y, ClassLabel")

def evaluate_and_save(clf, scaler, X_test, y_test, out_model=OUT_MODEL, out_fig=OUT_FIG, save_model_if_trained=False):
    # If a scaler is provided, transform test set
    X_for_pred = X_test
    if scaler is not None:
        try:
            X_for_pred = scaler.transform(X_test)
        except Exception as e:
            print("Warning: scaler.transform failed:", e)
            print("Proceeding with raw X_test.")

    # predict
    y_pred = clf.predict(X_for_pred)
    cm = confusion_matrix(y_test, y_pred)
    # normalized by true-class rows (avoid division by zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]

    # print results
    print("Confusion matrix (counts):")
    print(cm)
    print("\nConfusion matrix (normalized by true class rows):")
    print(np.round(cm_norm, 3))
    print("\nTP={}, FP={}, FN={}, TN={}".format(tp, fp, fn, tn))
    print("\nAccuracy = {:.4f}".format(acc))
    print("Precision = {:.4f}".format(prec))
    print("Recall = {:.4f}".format(rec))
    print("F1-score = {:.4f}".format(f1))

    # optionally save the model (only when we trained one here)
    if save_model_if_trained:
        try:
            joblib.dump(clf, out_model)
            print("\nSaved trained default model to:", out_model)
        except Exception as e:
            print("Could not save model:", e)

    # Plot predictions using original (unscaled) coordinates for visualization
    try:
        # If a scaler exists and we transformed X_test for prediction, we still want to plot the original X_test coords
        plt.figure(figsize=(5,4))
        plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='bwr', s=20, edgecolor='k', alpha=0.8)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.title('Test set predictions (blue=0 / red=1)')
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        plt.close()
        print("Saved prediction scatter to:", out_fig)
    except Exception as e:
        print("Could not save figure:", e)

def main():
    parser = argparse.ArgumentParser(description="Evaluate two-moons classifier on test CSV.")
    parser.add_argument('--test', help='Path to test CSV', default=TEST_CSV)
    parser.add_argument('--model', help='Path to saved model (joblib/pkl). If omitted, script trains a model.', default=None)
    parser.add_argument('--scaler', help='Path to saved scaler (joblib/pkl). Optional but recommended.', default=None)
    parser.add_argument('--save-trained', action='store_true', help='If training default model, save it to disk.')
    args = parser.parse_args()

    X_test, y_test = load_test_csv(args.test)

    clf = None
    scaler = None
    trained_here = False

    if args.model:
        if os.path.exists(args.model):
            try:
                clf = joblib.load(args.model)
                print("Loaded model from", args.model)
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{args.model}': {e}")
        else:
            raise FileNotFoundError("Model path {} not found".format(args.model))
    else:
        print("No model provided. Training a new MLP on synthetic two-moons data (this is only to enable evaluation).")
        clf, _ = train_default_model()
        trained_here = True

    # load scaler if provided
    if args.scaler:
        if os.path.exists(args.scaler):
            try:
                scaler = joblib.load(args.scaler)
                print("Loaded scaler from", args.scaler)
            except Exception as e:
                raise RuntimeError(f"Failed to load scaler '{args.scaler}': {e}")
        else:
            raise FileNotFoundError("Scaler path {} not found".format(args.scaler))
    else:
        # If no scaler provided, assume the model expects raw XY coordinates (or you saved scaler with model)
        scaler = None

    # If user provided a model that was trained on scaled inputs but didn't provide scaler,
    # warn them because results may be wrong
    if args.model and not args.scaler:
        print("Warning: You loaded a model but did not provide a scaler. Ensure this model expects raw (unscaled) inputs.")

    evaluate_and_save(clf, scaler, X_test, y_test, out_model=OUT_MODEL, out_fig=OUT_FIG, save_model_if_trained=args.save_trained)

if __name__ == "__main__":
    main()
