import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

classes = ['Early Stage', 'Mid Stage', 'Last Stage']

class_content = {
    'Early Stage': """EARLY STAGE:
• Sanitation: Trimming of the infected parts to stop the spread.
• Pesticides application: Apply every 7-14 days of interval.
Chemical treatment suggestions:
1. Copper oxychloride (synthetic chemical): 3g/L water.
2. Diphenaconozole (synthetic chemical): 0.5ml /L water.
3. Allium cepa extract (botanical chemical): 10ml /L water.

Important considerations during chemical treatment:
• Weather conditions: Ensure no rain for at least 24 hours for chemical adherence to the plant surfaces.
• Plant growth stages: Pay special attention when periods of rapid growth or when fruit is setting.
• Local regulations and recommendations: Consult local agricultural authorities for tailored advice and adhere to guidelines.
• Product label: Always follow the product label for concentration, frequency and safety precautions.
""",
    'Mid-Stage': """MID-STAGE:
• MID STAGE:
• Avoid over-irrigation: Be careful with the splashing of water. Drip or micro irrigation sprinkler is recommended.
• Pesticides application: Apply every 7days of interval.
Chemical treatment suggestions:
1. Copper oxychloride (synthetic chemical): 3g/L water.
2. Diphenaconozole (synthetic chemical): 0.5ml /L water.
3. Allium cepa extract (botanical chemical): 10ml /L water.

Important considerations during chemical treatment:
• Weather conditions: Ensure no rain for at least 24 hours for chemical adherence to the plant surfaces.
• Plant growth stages: Pay special attention when periods of rapid growth or when fruit is setting.
• Local regulations and recommendations: Consult local agricultural authorities for tailored advice and adhere to guidelines.
• Product label: Always follow the product label for concentration, frequency and safety precautions.
""",
    'Last Stage': """LAST STAGE:
• LATE STAGE:
• Tree removal: To prevent the spread to nearby healthy trees.
• Pesticides application: Apply every 5-7 days of interval.

Chemical treatment suggestions:
1. Copper oxychloride (synthetic chemical): 3g/L water.
2. Diphenaconozole (synthetic chemical): 0.5ml /L water.
3. Allium cepa extract (botanical chemical): 10ml /L water.

Important considerations during chemical treatment:
• Weather conditions: Ensure no rain for at least 24 hours for chemical adherence to the plant surfaces.
• Plant growth stages: Pay special attention when periods of rapid growth or when fruit is setting.
• Local regulations and recommendations: Consult local agricultural authorities for tailored advice and adhere to guidelines.
• Product label: Always follow the product label for concentration, frequency and safety precautions.
"""
}


st.title("Plant Stage Prediction")

@st.cache_resource
def load_model():
    if not os.path.exists("./model.h5"):
        print("Model Not Found. Downloading...")
        res = requests.get("https://drive.usercontent.google.com/download?id=1GJSck2sAQlMo_tVrDLTCUHP3_4VaOaoX&export=download")
        if res.status_code == 200:
            with open("./model.h5", "wb") as file:
                file.write(res.content)
        print("Model downloaded successfully!")
    else:
        print("Model Found.")
    print("Loading Model...")
    return tf.keras.models.load_model("./model.h5")

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
    pred_class_label = classes[pred_class_index]
    st.success(f"Predicted class probabilities: {pred}")
    st.success("Predicted class: " + pred_class_label)
    if pred_class_label in class_content:
        st.success(class_content[pred_class_label])
    else:
        st.success("No content available for predicted classs:", pred_class_label)