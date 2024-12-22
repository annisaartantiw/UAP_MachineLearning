import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

# Fungsi untuk menyisipkan CSS kustom dengan efek blur hanya pada background
def add_background_with_blur(image_file, blur_level=4):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: none;
        }}
        .background {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("data:image/jpeg;base64,{image_file}") no-repeat center center fixed;
            background-size: cover;
            filter: blur({blur_level}px);
            z-index: -1;
        }}
        </style>
        <div class="background"></div>
        """,
        unsafe_allow_html=True
    )

# Konversi file gambar menjadi base64
with open("src/images/background.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Tambahkan latar belakang dengan tingkat blur tetap
add_background_with_blur(encoded_image, blur_level=4)

# Set title for the web app
st.title("Image Classification with Deep Learning")
st.write("Upload an image to classify it using our pre-trained models.")

# Load models
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model_cnn_path = 'src/model/model_cnn_final.keras'  # Update with your file path
model_vgg_path = 'src/model/model_vgg_final.keras'  # Update with your file path

model_cnn = load_model(model_cnn_path)
model_vgg = load_model(model_vgg_path)

# Define class names
class_names = ['apple_6', 'apple_braeburn_1', 'apple_crimson_snow_1', 
               'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 
               'apple_granny_smith_1', 'apple_hit_1', 'apple_pink_lady_1', 
               'apple_red_1', 'apple_red_2', 'apple_red_3', 
               'apple_red_delicios_1', 'apple_red_yellow_1', 'apple_rotten_1', 
               'cabbage_white_1', 'carrot_1', 'cucumber_1', 'cucumber_3', 
               'eggplant_long_1', 'pear_1', 'pear_3', 'zucchini_1', 
               'zucchini_dark_1']

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Load and preprocess the image
    img = Image.open(uploaded_file).resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Select model
    model_choice = st.selectbox("Select a model for classification:", ("CNN", "VGG"))

    if model_choice == "CNN":
        model = model_cnn
    else:
        model = model_vgg

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display prediction
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")

    # Visualize prediction confidence levels
    st.write("### Confidence Levels:")
    st.bar_chart(predictions[0])
    st.write("Class Labels:")
    st.write(", ".join(class_names))
    
st.write("\n---\nDeveloped by Artanti")
