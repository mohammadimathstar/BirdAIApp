import os
import io
import pickle
import requests
import yaml


from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms

from src.AChorDSLVQ.model import Model
from src.explainability.visualize_prediction import compute_region_importance, compute_pixel_importance


# ------------------------------
# Setup Directories and Logger
# ------------------------------
STATIC_DIR = "static"
ORIGINALS_DIR = os.path.join(STATIC_DIR, "originals")
HEATMAPS_DIR = os.path.join(STATIC_DIR, "heatmaps")
os.makedirs(ORIGINALS_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)


# ------------------------------
# Image Transformation
# ------------------------------
def transform_image(image_bytes):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


# ------------------------------
# Read Image as Bytes
# ------------------------------
def get_bytes_from_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()



# ------------------------------
# Download and Save Image
# ------------------------------
def download_image(url, logger):
    try:
        response = requests.get(url)
        response.raise_for_status()

        filename = os.path.basename(url).split("?")[0]
        save_path = os.path.join(ORIGINALS_DIR, filename)

        with open(save_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Image downloaded and saved at '{save_path}'")
        return save_path
    except requests.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise



# ------------------------------
# Resize Image and Save
# ------------------------------
def resize_and_save_image(image_path, logger):
    try:
        img_org = Image.open(image_path)
        img_cv2 = cv2.cvtColor(np.array(img_org), cv2.COLOR_RGB2BGR)
        resized_img = cv2.resize(img_cv2, (224, 224), interpolation=cv2.INTER_LINEAR)

        resized_filename = os.path.basename(image_path).replace(".jpg", "_resized.jpg")
        resized_path = os.path.join(ORIGINALS_DIR, resized_filename)

        plt.imsave(resized_path, resized_img, vmin=0.0, vmax=1.0)
        return resized_path, resized_img

    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        raise


# ------------------------------
# Generate Heatmap
# ------------------------------
def generate_heatmap(img_tensor, img_resized_array, logger):
    try:
        region_importance = compute_region_importance(model, img_tensor)
        compute_pixel_importance(img_resized_array, region_importance, HEATMAPS_DIR)

        heatmap_path = os.path.join(HEATMAPS_DIR, 'heatmap_original_image.png')
        logger.info(f"Heatmap generated at {heatmap_path}")
        return heatmap_path
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")
        raise


# ------------------------------
# Model Prediction
# ------------------------------
@torch.no_grad()
def get_prediction(img):
    model.eval()
    distances, _ = model(img)
    ypred = model.prototype_layer.yprotos[distances.argmin(axis=1)].item()
    if index2class:
        assert ypred in index2class.keys(), f"Prediction {ypred} not in class labels."
        output = {'prediction': ypred, 'class': index2class.get(ypred).replace("_", " ")}
    else:
        output = {'prediction': ypred}
    return output


# ------------------------------
# Load Class Labels and Model
# ------------------------------
with open("params.yaml", 'r') as f:
    config = yaml.safe_load(f)
index2class_path = 'index2labels/index2label_%s.pkl' % config['data_loader']['data_name']
with open(index2class_path, 'rb') as f:
    index2class = pickle.load(f)


model = Model.load(config['explainability']['model_path'])
