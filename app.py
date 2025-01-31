import os
from flask import Flask, request, jsonify, render_template
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
from utils import get_prediction, get_bytes_from_image, download_image, generate_heatmap, transform_image
from utils import resize_and_save_image
import time


app = Flask(__name__)

# ------------------------------
# Logging Configuration
# ------------------------------
logger = logging.getLogger("BirdAPI")
logger.setLevel(logging.INFO)

fomatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(
    filename='./predictor.log', maxBytes=5000, backupCount=1
)
file_handler.setFormatter(fomatter)
logger.addHandler(file_handler)


# ------------------------------
# Directory Setup
# ------------------------------
STATIC_DIR = "static"
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
ORIGINALS_DIR = os.path.join(STATIC_DIR, "originals")
HEATMAPS_DIR = os.path.join(STATIC_DIR, "heatmaps")

for directory in [UPLOADS_DIR, ORIGINALS_DIR, HEATMAPS_DIR]:
    os.makedirs(directory, exist_ok=True)


# ------------------------------
# Home Route (Image Upload & Prediction)
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        image_url = request.form.get("image_url")
        uploaded_file = request.files.get("image_file")
        filename = None

        try:
            if uploaded_file and uploaded_file.filename != '':
                # Save uploaded image
                filename = f"{int(time.time())}_{uploaded_file.filename}"
                original_image_path = os.path.join(UPLOADS_DIR, filename)
                uploaded_file.save(original_image_path)
                img = Image.open(original_image_path)
                img_url = f"/{original_image_path}"
            elif image_url:
                # Download and save the image from URL
                original_image_path = download_image(image_url, logger)
                img_url = f"/{original_image_path}"
            else:
                return render_template("index.html", error="No image uploaded or URL provided")


            # Transform the image for the model
            img_bytes = get_bytes_from_image(original_image_path)
            img_tensor = transform_image(img_bytes)

            # Resize and save a copy of the original image
            resized_image_path, resized_image_array = resize_and_save_image(original_image_path, logger)

            # Perform prediction
            result = get_prediction(img_tensor)
            logger.info(f"Prediction: '{result['class']}'")


            # Generate and save heatmap
            heatmap_path = generate_heatmap(img_tensor, resized_image_array, logger)

            return render_template("result.html",
                                   image_url=resized_image_path,
                                   heatmap_url=heatmap_path,
                                   prediction=result['class'])
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return render_template("index.html", error="Error processing image")

    return render_template("index.html")






# curl --location 'http://127.0.0.1:5000/predict' --form 'image_url="https://media.audubon.org/nas_birdapi_hero/h_a1_7508_2_indigo-bunting_julie_torkomian_breeding-adult-male.jpg"'
# curl --location 'http://127.0.0.1:5000/predict' --form 'image_url="https://indianaaudubon.org/wp-content/uploads/2016/04/Cardinal_Northern_male_Ash_2012.jpg"'

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))