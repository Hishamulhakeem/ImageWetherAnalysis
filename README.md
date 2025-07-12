# Weather Image Classification 

This project demonstrates how to classify weather-related images (e.g., sunny, rainy, cloudy) using image embeddings generated with `img2vec-pytorch` and a Random Forest Classifier.

## 📂 Project Structure

- `main.py` — Trains a Random Forest model on image embeddings.
- `infy.py` — Loads the trained model and predicts the weather type for a test image.
- `model.p` — The saved trained model (generated after running `main.py`).
- `./data/wether_dataset/` — Folder containing the training and test images organized by class.

## 🔍 How It Works

1. **Image Embeddings:** Converts input images into feature vectors using pretrained models via `img2vec-pytorch`.
2. **Model Training:** Trains a Random Forest Classifier on these vectors to learn different weather conditions.
3. **Prediction:** Takes a new image, converts it to vector form, predicts its label, and displays the result using OpenCV.

## 📦 Requirements

Install dependencies using:

```bash
pip install img2vec-pytorch scikit-learn opencv-python pillow

Make sure your dataset is placed in the following structure:

data/
└── wether_dataset/
    ├── train/
    │   ├── sunny/
    │   ├── rainy/
    │   └── ...
    └── val/
        ├── sunny/
        ├── rainy/
        └── ...

