import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

classes = ['Early Stage', 'Mid Stage', 'Last Stage']

st.title("Plant Stage Prediction")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

def preprocess_image(img):
    img = image.load_img(img, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array


model = load_model()

img = st.file_uploader(label="Image File", type=(['png', 'jpg']))


if img:
    st.image(img)
    img_arr = preprocess_image(img)
    print(img_arr)
    pred = model.predict(img_arr)
    pred_class_index = np.argmax(pred)
    st.success(f"Predicted class probabilities: {pred}")
    st.success("Predicted class: " + classes[pred_class_index])
