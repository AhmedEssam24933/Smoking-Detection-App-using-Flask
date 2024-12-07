from flask import Flask, request, jsonify
import os
import pathlib
import shutil
import torch
import cv2
import warnings
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import logging

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Set model path and initialize the model
os.chdir(r'E:\Fassla Projects\Smoking Detection Model App\Yolo Model')
pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('.', 'custom', path=r'E:\Fassla Projects\Smoking Detection Model App\Yolo Model\Smoking Detection3.pt', source='local', force_reload=True)

# Define paths
base_detect_path = r"runs\detect"

# Helper function to delete folder or file
def delete_folder_or_file(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"Deleted folder or file at path: {path}")
            return "Deletion completed successfully"
        else:
            logging.warning(f"Path does not exist: {path}")
            return "Path does not exist"
    except FileNotFoundError as e:
        logging.error(f"Error occurred while deleting the file/folder: {e}")
        return f"Error occurred while deleting the file/folder: {e}"

# Apply detection model and check results
def apply_detection_model_and_check(model, image):
    try:
        result = model(image)  # Perform detection
        result.crop(save=True)  # Crop the detected regions and save

        # Check if "Smoking" folder exists and assess confidence score
        for folder in os.listdir(base_detect_path):
            folder_path = os.path.join(base_detect_path, folder, "crops", "Smooking")
            if os.path.exists(folder_path):
                # Check the confusion score (assuming it's stored in result.pred)
                confidences = result.xywh[0][:, 4].cpu().numpy()  # Extract the confidence scores (for each detection)
                print(confidences)
                logging.info(f"Confusion score: {confidences}")
                if confidences > 0.8:
                    logging.info("Smoking detected in the image with high confidence.")
                    return True
                else:
                    logging.info(f"Smoking detected, but confidence ({confidences}) is below the threshold. Continuing...")
                    continue  # Continue checking next folders if confusion is low

        # If no smoking is detected with high confidence in any folder
        logging.info("No smoking detected in any image.")
        return False
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return False

# Download image from URL and convert to OpenCV format
def download_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            logging.warning(f"Failed to download image from {url}")
            return None
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

@app.route('/api/detect', methods=['POST'])
def detect():
    # Check if the request contains a valid JSON payload
    if not request.is_json:
        return jsonify({"error": "Invalid input, JSON expected"}), 400

    data = request.get_json()

    # Validate the input JSON structure
    if 'post_id' not in data or 'images_links' not in data:
        return jsonify({"error": "Invalid input, 'post_id' and 'image_links' are required"}), 400

    post_id = data['post_id']
    image_links = data['images_links']

    # Ensure `image_links` is either a list or a single string
    if not isinstance(image_links, (list, str)):
        return jsonify({"error": "'images_links' should be a string (for one image) or a list"}), 400

    # Normalize `image_links` to a list
    if isinstance(image_links, str):
        image_links = [image_links]

    # Delete any previous results
    delete_folder_or_file(base_detect_path)

    for image_url in image_links:
        try:
            # Download image from the URL
            image = download_image_from_url(image_url)
            if image is None:
                continue

            # Apply the detection model
            if apply_detection_model_and_check(model, image):
                # Notify the Node.js server
                url = "http://localhost:8000/post/images"
                headers = {
                    'Content-Type': 'application/json',
                    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTgzNywicm9sZSI6IlNUVURFTlQiLCJpYXQiOjE3MzMzMTExMzEsImV4cCI6MTczNDUyMDczMX0.8FOdsBZLvnwcFkpWSU3uo0gWDGhKaNSJcI3nYahCE7c"
                }
                result = {'postId': post_id}
                try:
                    response = requests.delete(url, json=result, headers=headers)
                    if response.status_code == 200:
                        logging.info("Notification to Node.js server was successful!")
                    else:
                        logging.error(f"Failed with status code {response.status_code}")
                        logging.error("Response: %s", response.text)
                except requests.RequestException as e:
                    logging.error("Error in sending request to external API: %s", e)

                # Return only the post_id and stop processing further
                return jsonify({"post_id": post_id})
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            continue

    # If no smoking detected in all images, return no content
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)











