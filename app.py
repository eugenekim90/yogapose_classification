import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# Load class labels from JSON
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
    # Ensure class names are in order of index
    class_names = [class_labels[str(i)] for i in range(len(class_labels))]

# Model options and mapping
MODEL_DIR = 'models'
MODEL_FILES = {
    'CNN Transfer Learning (EfficientNetB0, Quantized)': 'tl_model_quant.tflite',
    'Custom CNN (Quantized)': 'advanced_model_quant.tflite'
}

st.title('Yoga Pose Classifier')
st.write('Upload a yoga pose image and select a model to predict the pose.')

# Helper to get model path
def get_model_path(model_display_name):
    return os.path.join(MODEL_DIR, MODEL_FILES[model_display_name])

# Model selector
model_display_name = st.selectbox('Select Model', list(MODEL_FILES.keys()))

# Image uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # --- Preprocessing ---
    # Always resize to (224, 224) for both models
    img_resized = image.resize((224, 224))
    if model_display_name == 'CNN Transfer Learning (EfficientNetB0, Quantized)':
        # EfficientNet: resize only, no normalization
        img_array = np.array(img_resized, dtype=np.float32)
    else:
        # Custom CNN: resize and normalize to [0, 1]
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # --- Model Inference ---
    interpreter = tf.lite.Interpreter(model_path=get_model_path(model_display_name))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor and run inference
    input_data = img_array.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class
    pred_class = int(np.argmax(output))
    pred_label = class_names[pred_class]
    st.success(f'Predicted Pose: {pred_label}') 