### name : Sukhdeep Singh - 02210861
# main.py

import os
import random
from glob import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ====== CONFIG ======
DATASET_PATH = "./HeatMaps"   # change if your folder name is different
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_SEED = 42
# ====================


def get_model_folders(root_path):
    """Return a sorted list of model folder names inside DATASET_PATH."""
    model_names = [
        d for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]
    model_names.sort()
    return model_names


class ImageSequence:
    """
    Represents one 'video' = one numbered subfolder
    containing 5 heatmap PNGs.
    """

    def __init__(self, folder_path, label_index, class_name, pull_frames=False):
        self.folder_path = folder_path
        self.label_index = label_index
        self.class_name = class_name
        self._frames = None
        if pull_frames:
            _ = self.frames

    @property
    def frames(self):
        """Lazy-load all PNG images in this folder."""
        if self._frames is None:
            self._frames = self.load_frames()
        return self._frames

    def load_frames(self):
        # Get all .png files, sorted
        img_paths = sorted(glob(os.path.join(self.folder_path, "*.png")))
        frames = []
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")  # ensure 3 channels
                frames.append(img)
            except Exception as e:
                print(f"Could not load image {p}: {e}")
        return frames


def extract_features_from_image(img: Image.Image, bins_per_channel=16):
    """
    Simple, classic feature: color histogram for each channel (R,G,B).
    No CNN, just counting colors with numpy.
    """
    arr = np.asarray(img)  # shape: (H, W, 3)

    # Split channels
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    hist_r, _ = np.histogram(r, bins=bins_per_channel, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=bins_per_channel, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=bins_per_channel, range=(0, 256))

    hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

    # Normalize
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist  # length = bins_per_channel * 3


def extract_features_from_sequence(seq: ImageSequence):
    """
    Extract features for all frames in a sequence
    and average them to get one feature vector per 'video'.
    """
    frames = seq.frames
    feature_list = []
    for img in frames:
        feat = extract_features_from_image(img)
        feature_list.append(feat)

    if not feature_list:
        # In weird case of empty folder; return zeros
        return np.zeros(16 * 3, dtype=np.float32)

    features = np.stack(feature_list, axis=0)  # shape: (num_frames, feat_dim)
    return features.mean(axis=0)  # average over frames


def build_dataset(root_path):
    """
    Walk the folder structure and build:
    X: feature matrix (n_samples, n_features)
    y: labels (n_samples,)
    model_names: list mapping label -> model name
    sample_ids: list of strings like "ModelName/42"
    """
    random.seed(RANDOM_SEED)

    X = []
    y = []
    sample_ids = []

    model_names = get_model_folders(root_path)

    for label_index, model_name in enumerate(model_names):
        model_root = os.path.join(root_path, model_name)

        # Each numbered subfolder is one 'video'
        subfolders = [
            d for d in os.listdir(model_root)
            if os.path.isdir(os.path.join(model_root, d))
        ]

        for folder_name in subfolders:
            folder_path = os.path.join(model_root, folder_name)

            seq = ImageSequence(folder_path, label_index, model_name)
            feat_vec = extract_features_from_sequence(seq)

            X.append(feat_vec)
            y.append(label_index)
            sample_ids.append(f"{model_name}/{folder_name}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, model_names, sample_ids


def split_dataset(X, y):
    """
    Split into train / val / test using sklearn.
    Uses stratification so each split has all classes.
    """
    # First: split off TEST
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Now split TEMP into TRAIN and VAL
    # (VAL_SIZE is relative to full set; adjust for smaller temp)
    val_fraction_of_temp = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train):
    """
    Train a classic ML model.
    RandomForest works well on histogram-style features.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X, y, split_name, label_names):
    """
    Print accuracy and a classification report for a given split.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n==== {split_name.upper()} RESULTS ====")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=label_names))

    print("Confusion matrix (rows: true, cols: predicted):")
    print(confusion_matrix(y, y_pred))


def main():
    print("Building dataset from:", DATASET_PATH)
    X, y, model_names, sample_ids = build_dataset(DATASET_PATH)
    print("Total samples:", X.shape[0])
    print("Feature dimension:", X.shape[1])
    print("Model labels:", {i: name for i, name in enumerate(model_names)})

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print("\nSplit sizes:")
    print("Train:", X_train.shape[0],
          " Val:", X_val.shape[0],
          " Test:", X_test.shape[0])

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    # Evaluate on all splits
    evaluate_model(model, X_train, y_train, "train", model_names)
    evaluate_model(model, X_val, y_val, "validation", model_names)
    evaluate_model(model, X_test, y_test, "test", model_names)

    # Save the model to disk for later use
    import joblib
    joblib.dump(model, "model_random_forest.pkl")
    print("\nSaved trained model to model_random_forest.pkl")


if __name__ == "__main__":
    main()
