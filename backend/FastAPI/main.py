# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:46:29 2024
"""

import uvicorn
from fastapi import FastAPI
from DiseaseImage import DiseaseImage 

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import time
import h5py
from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


####################################################

# Google Drive File ID
DRIVE_FILE_ID = "1-49L8zFLvPiqW80XRx1RfTSTkgiRWv7G"  # Replace with your Google Drive File ID

# Define the model directory path and load the trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_PATH = f"{working_dir}/trained_model/plant_disease_prediction_model.h5" # Temporary cache path


# Function to download model from Google Drive
# Download Model with Retry and Error Handling
def download_model_from_drive(file_id, destination, retries=3):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(destination, "wb") as f:
                    f.write(response.content)
                print(f"Model downloaded successfully to {destination}")
                return
            else:
                print(f"Attempt {attempt + 1}: Failed to download model (status code {response.status_code})")
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error downloading model:", e)
    raise Exception("Failed to download model after multiple attempts.")

# Download and load the model
if not os.path.exists(MODEL_CACHE_PATH):
    print("Downloading model from Google Drive...")
    download_model_from_drive(DRIVE_FILE_ID, MODEL_CACHE_PATH)

# Validate the downloaded file
if os.path.exists(MODEL_CACHE_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_CACHE_PATH)
        print("Model loaded successfully.")
    except OSError as e:
        raise RuntimeError("Model loading failed. Please check the downloaded file.") from e
else:
    raise FileNotFoundError("Model file not found. Download may have failed.")


####################################################


# Load the model
model = tf.keras.models.load_model(MODEL_CACHE_PATH)

# Load class indices
class_indices = json.load(open(f"{working_dir}/trained_model/class_indices.json"))

def load_and_preprocess_image(image_path_or_url, target_size=(224, 224)):
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path_or_url)

    img = img.resize(target_size)
    
    img_array = np.array(img)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def predict_image_class(model, image_path_or_url, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path_or_url)
    
    predictions = model.predict(preprocessed_img)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    return predicted_class_name

@app.get('/')
def index():
    return {'message': 'Hello, the Plant Disease Detection API is up and running!'}

@app.post('/detect')
def detect_disease(data: DiseaseImage):
    image_url = data.image
    
    prediction = predict_image_class(model, image_url, class_indices)
    
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
