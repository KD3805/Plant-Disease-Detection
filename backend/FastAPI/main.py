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


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

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
    uvicorn.run(app, host='127.0.0.1', port=8000)

