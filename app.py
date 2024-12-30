import os
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from io import BytesIO
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model using tf.keras
model = None

try:
    model = tf.keras.models.load_model('my_model.keras')  # Replace with your model path (e.g. .h5 or .keras)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Class labels (same as in your model training)
class_names = {
    0: 'no_DR',
    1: 'mild_DR',
    2: 'moderate_DR',
    3: 'severe_DR',
    4: 'proliferative_DR'
}

# Authentication Key Dependency
API_KEY = "nawabBhaikamodel"

def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

# Image preprocessing function
async def preprocess_image(file: UploadFile):
    try:
        # Read image from memory as bytes
        image_bytes = BytesIO(await file.read())
        
        # Use OpenCV to decode the image
        img = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image file.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224, 224))  # Resize to the required size for the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize the image
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Predict route with authentication
@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    if model is None:
        return {"error": "Model not loaded."}

    # Preprocess the image without saving it
    img = await preprocess_image(file)
    
    if img is None:
        return {"error": "Could not process the image."}

    try:
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]
        confidence = prediction[0][predicted_class] * 100

        # Return prediction results
        return {
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
        }
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Prediction failed."}

# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
