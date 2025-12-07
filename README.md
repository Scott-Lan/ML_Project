# ML_Project
# AI Video Model Classifier (Heatmap-Based, Classic ML)

This project trains a **classic machine learning model** (no CNNs) to predict **which AI video generation model** produced a given video, using only **5 heatmap frames** per video.

It was developed for a Machine Learning course project where the original goal was to detect AI-generated video. As a more scoped and achievable version, we focus on **classifying the source model** among several AI video generators.

---

## 1. Project Overview

**Goal:**  
Given 5 heatmap images from a video, predict **which AI model** generated that video.

**Key points:**

- Each **video** is represented by a folder containing **5 PNG heatmap frames**.
- There are multiple **generation models**, each with its own folder:
  - `BDAnimateDiffLightning`
  - `CogVideoX5B`
  - `RunwayML`
  - `StableDiffusion`
  - `Veo`
  - `VideoPoet`
- For each “video folder”, we:
  1. Load the 5 heatmap images.
  2. Compute **color histograms** per image (R,G,B channels, 16 bins each → 48 features).
  3. Average features across the 5 frames → one 48-D feature vector per video.
- We train a **Random Forest classifier** to predict the model label from these features.
- Train / validation / test split: **80% / 10% / 10%** with stratification.

On the provided dataset (600 samples, 6 classes), we observe around **75% test accuracy**, well above random chance (≈16.7%).

---

## 2. Folder Structure

Expected project layout:

```text
ML_Project/
├── HeatMaps/
│   ├── BDAnimateDiffLightning/
│   │   ├── 0/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   ├── 2.png
│   │   │   ├── 3.png
│   │   │   └── 4.png
│   │   ├── 1/
│   │   ├── ...
│   ├── CogVideoX5B/
│   ├── RunwayML/
│   ├── StableDiffusion/
│   ├── Veo/
│   └── VideoPoet/
├── main.py
├── predict_one.py
├── test_new_dataset.py  (optional)
└── README.md
