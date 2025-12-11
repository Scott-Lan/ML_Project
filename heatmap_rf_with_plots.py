import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# ------------------------
# CONFIG
# ------------------------

HEATMAP_ROOT = "./HeatMaps"   # folder containing subfolders per model
BINS_PER_CHANNEL = 16         # 16 R + 16 G + 16 B = 48 features
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_SEED = 42
N_TREES = 200                  # number of trees in the RandomForest

# ------------------------
# FEATURE EXTRACTION
# ------------------------

def extract_hist_features(img_path, bins_per_channel=BINS_PER_CHANNEL):
    """
    Extract a 48-dimensional normalized RGB histogram feature from an image.
    16 bins for each channel (R, G, B), concatenated and normalized.
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    # Split channels
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    hist_r, _ = np.histogram(r, bins=bins_per_channel, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=bins_per_channel, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=bins_per_channel, range=(0, 256))

    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

    # Normalize to fractions (sum = 1)
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist  # shape (48,)


def build_dataset(root_dir=HEATMAP_ROOT):
    """
    Build X (features) and y (labels) from the HeatMaps folder.

    Assumes structure like:
        HeatMaps/
            BDAnimateDiffLightning/
                0/   (folder with 5 PNGs)
                1/
                ...
            CogVideoX5B/
                0/
                ...
            ...

    Each numbered subfolder is treated as one sample (one video).
    We load all PNGs in that folder, compute 48-dim features per image,
    then average them to get a single 48-dim vector per video.
    """
    X = []
    y = []
    class_names = []

    # iterate over model folders (e.g., BDAnimateDiffLightning, CogVideoX5B, etc.)
    for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_names.append(class_name)
        print(f"Processing class {class_idx}: {class_name}")

        # each numbered folder inside is a "video" sample
        for sample_name in sorted(os.listdir(class_path)):
            sample_path = os.path.join(class_path, sample_name)
            if not os.path.isdir(sample_path):
                continue

            # collect all .png files in this sample folder
            image_files = [
                f for f in os.listdir(sample_path)
                if f.lower().endswith(".png")
            ]
            if len(image_files) == 0:
                continue

            features_list = []
            for img_file in image_files:
                img_path = os.path.join(sample_path, img_file)
                feat = extract_hist_features(img_path)
                features_list.append(feat)

            # average features over all heatmaps in this video
            features_array = np.stack(features_list, axis=0)  # (num_frames, 48)
            video_feat = features_array.mean(axis=0)          # (48,)

            X.append(video_feat)
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, class_names


# ------------------------
# PLOTTING HELPERS
# ------------------------

def plot_confusion_matrix(cm, class_names, filename):
    fig, ax = plt.subplots(figsize=(7, 7))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=True, values_format="d")

    # Make x-axis labels nicely aligned under each column
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Random Forest – Test Confusion Matrix")

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix to {filename}")



def plot_per_class_f1(report_dict, class_names, filename):
    """
    report_dict is from classification_report(..., output_dict=True)
    """
    f1_scores = [report_dict[name]["f1-score"] for name in class_names]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(class_names))
    plt.bar(x, f1_scores)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("F1-score")
    plt.title("Random Forest – Per-Class F1 (Test Set)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved per-class F1 bar chart to {filename}")


def plot_feature_importance(model, filename):
    """
    Plot the top 15 most important histogram features.
    """
    importances = model.feature_importances_
    n_features = len(importances)

    # create feature names: R0..R15, G0..G15, B0..B15
    names = []
    bins = BINS_PER_CHANNEL
    for ch in ["R", "G", "B"]:
        for i in range(bins):
            names.append(f"{ch}{i}")

    # just in case
    if len(names) != n_features:
        names = [f"feat_{i}" for i in range(n_features)]

    # sort by importance
    idx_sorted = np.argsort(importances)[::-1]  # descending
    top_k = 15
    top_idx = idx_sorted[:top_k]
    top_importances = importances[top_idx]
    top_names = [names[i] for i in top_idx]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(top_names))
    plt.bar(x, top_importances)
    plt.xticks(x, top_names, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Random Forest – Top Feature Importances")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to {filename}")


# ------------------------
# MAIN SCRIPT
# ------------------------

def main():
    np.random.seed(RANDOM_SEED)

    print(f"Building dataset from: {HEATMAP_ROOT}")
    X, y, class_names = build_dataset(HEATMAP_ROOT)
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print("Model labels:", {i: name for i, name in enumerate(class_names)})

    # Split into train/val/test: first test, then val from remaining
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # adjust val size relative to remaining
    val_ratio_adjusted = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print("\nSplit sizes:")
    print("Train:", len(X_train), " Val:", len(X_val), " Test:", len(X_test))

    # Train Random Forest
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate helper
    def eval_split(name, X_split, y_split):
        print(f"\n==== {name.upper()} RESULTS ====")
        y_pred = clf.predict(X_split)
        report = classification_report(
            y_split, y_pred, target_names=class_names, digits=4
        )
        print("Classification report:\n", report)
        cm = confusion_matrix(y_split, y_pred)
        print("Confusion matrix (rows: true, cols: predicted):")
        print(cm)
        return y_pred, report, cm

    # Train/Val/Test evaluation
    y_train_pred, report_train, cm_train = eval_split("train", X_train, y_train)
    y_val_pred, report_val, cm_val = eval_split("validation", X_val, y_val)
    y_test_pred, report_test, cm_test = eval_split("test", X_test, y_test)

    # Save model
    model_path = "model_random_forest_heatmaps.pkl"
    joblib.dump(clf, model_path)
    print(f"\nSaved trained model to {model_path}")

    # --- Plotting for TEST SET ---

    # 1) Confusion matrix
    plot_confusion_matrix(cm_test, class_names, "rf_confusion_matrix_test.png")

    # 2) Per-class F1 scores
    report_dict_test = classification_report(
        y_test, y_test_pred, target_names=class_names, output_dict=True
    )
    plot_per_class_f1(report_dict_test, class_names, "rf_per_class_f1_test.png")

    # 3) Feature importance
    plot_feature_importance(clf, "rf_feature_importance.png")


if __name__ == "__main__":
    main()
