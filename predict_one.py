# predict_one.py
### name: Sukhdeep Singh - 02210861
import os
from glob import glob
import numpy as np
from PIL import Image
import joblib

from main import extract_features_from_image  # reuse function


def extract_features_from_folder(folder_path):
    img_paths = sorted(glob(os.path.join(folder_path, "*.png")))
    feature_list = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        feat = extract_features_from_image(img)
        feature_list.append(feat)

    if not feature_list:
        return np.zeros(16 * 3, dtype=np.float32)

    features = np.stack(feature_list, axis=0)
    return features.mean(axis=0)


def main():
    # path to a folder like "HeatMaps/BDAnimateDiffLightning/42"
    folder_path = input("Enter path to video folder: ").strip()

    model = joblib.load("model_random_forest.pkl")

    # same order of model names as in main.py
    model_names = sorted(
        d for d in os.listdir("HeatMaps")
        if os.path.isdir(os.path.join("HeatMaps", d))
    )

    feat_vec = extract_features_from_folder(folder_path)
    feat_vec = feat_vec.reshape(1, -1)

    pred_label = model.predict(feat_vec)[0]
    print("Predicted model:", model_names[pred_label])


if __name__ == "__main__":
    main()
