"""
Streamlit app for crop disease detection.
Upload leaf image → preprocess → predict → show class, confidence, top-3 alternatives.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Configuration
CONFIG = {
    'model_path': 'best_model.keras',
    'class_names_path': 'class_names.json',
    'image_size': (224, 224),
    'confidence_threshold': 0.3
}


@st.cache_resource
def load_model():
    """Load trained model with caching."""
    try:
        model = tf.keras.models.load_model(CONFIG['model_path'])
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_class_names():
    """Load class names from JSON file."""
    try:
        with open(CONFIG['class_names_path'], 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        st.error(f"Class names file not found: {CONFIG['class_names_path']}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid JSON format in class names file")
        return None


def preprocess_image(image):
    """Preprocess uploaded image to match training pipeline."""
    image = image.resize(CONFIG['image_size'])
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_disease(model, image_array, class_names):
    """Predict disease class and return top predictions."""
    predictions = model.predict(image_array, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            'class': class_names[i],
            'confidence': float(predictions[i])
        }
        for i in top_indices
    ]
    
    return top_predictions


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Crop Disease Detection",
        page_icon="🌱",
        layout="centered"
    )
    
    st.title("🌱 Crop Disease Detection")
    st.write("Upload a leaf image to detect potential diseases")
    
    model = load_model()
    class_names = load_class_names()
    
    if model is None or class_names is None:
        st.error("Application cannot start. Please ensure model and class names files exist.")
        st.stop()
    
    uploaded_file = st.file_uploader(
        "Choose a leaf image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.write("Processing...")
                
                image_array = preprocess_image(image)
                predictions = predict_disease(model, image_array, class_names)
                
                top_prediction = predictions[0]
                
                st.subheader("Prediction")
                st.success(f"**{top_prediction['class']}**")
                st.metric("Confidence", f"{top_prediction['confidence']:.2%}")
                
                if top_prediction['confidence'] < CONFIG['confidence_threshold']:
                    st.warning(
                        f"⚠️ Low confidence ({top_prediction['confidence']:.2%}). "
                        "Image may not be a valid leaf or disease not in training set."
                    )
                
                st.subheader("Top 3 Alternatives")
                for i, pred in enumerate(predictions, 1):
                    st.write(f"{i}. {pred['class']} - {pred['confidence']:.2%}")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    st.divider()
    st.write("### Instructions")
    st.write("1. Upload a clear image of a plant leaf")
    st.write("2. The model will predict the disease class")
    st.write("3. Check confidence score - low scores indicate uncertain predictions")


if __name__ == '__main__':
    main()
