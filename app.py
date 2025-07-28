import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from mtcnn.mtcnn import MTCNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
print(tf.__version__)

# -------------------------------
# Sample dataset simulation
# -------------------------------
@st.cache_data
def load_dataset():
    # Simulated data for demo
    data = {
        "Image_ID": [f"img_{i}" for i in range(1, 11)],
        "Face_Count": np.random.randint(1, 5, 10),
        "Resolution": ["224x224"] * 10
    }
    df = pd.DataFrame(data)
    return df



# -------------------------------
# Dummy feature extraction
# -------------------------------
def extract_features(df):
    df['has_multiple_faces'] = df['Face_Count'].apply(lambda x: 1 if x > 2 else 0)
    X = df[['Face_Count']]
    y = df['has_multiple_faces']
    return X, y

# -------------------------------
# Face Detection
# -------------------------------
def detect_faces(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return image, len(results)

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Human Face Detection", layout="wide")
st.title("👤 Human Faces (Object Detection)")
menu = st.sidebar.radio("Select Menu", ["1️⃣ Data", "2️⃣ EDA - Visual", "3️⃣ Prediction"])

# -------------------------------
# 1. DATA SECTION
# -------------------------------
if menu.startswith("1"):
    st.subheader("📁 Dataset Overview")
    df = load_dataset()
    st.write("### Sample Dataset")
    st.dataframe(df)

    st.write("### Simulated Model Performance Data")
    perf_data = {
        "Metric": ["Precision", "Recall", "Accuracy", "F1 Score"],
        "Value (%)": [88, 87, 89, 86]
    }
    st.table(pd.DataFrame(perf_data))

# -------------------------------
# 2. EDA SECTION
# -------------------------------
elif menu.startswith("2"):
    st.subheader("📊 Exploratory Data Analysis")

    df = load_dataset()
    st.write("### Face Count Distribution")
    fig1 = px.histogram(df, x="Face_Count", nbins=5, title="Faces per Image")
    st.plotly_chart(fig1)

    st.write("### Image Resolution Distribution")
    fig2 = px.pie(df, names="Resolution", title="Image Resolution")
    st.plotly_chart(fig2)

# -------------------------------
# 3. PREDICTION SECTION
# -------------------------------
elif menu.startswith("3"):
    st.subheader("🔍 Face Detection & Prediction")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

        result_img, face_count = detect_faces(image_rgb)
        st.image(result_img, caption=f"Detected {face_count} face(s)", use_column_width=True)

        st.success(f"Detected {face_count} face(s) in the image")

    # Dummy classifier for demo
    st.write("### Predict if Image Has Multiple Faces (Demo Model)")

    df = load_dataset()
    X, y = extract_features(df)
    model = RandomForestClassifier().fit(X, y)
    input_faces = st.number_input("Enter number of faces detected", min_value=0, step=1)
    if st.button("Predict"):
        prediction = model.predict([[input_faces]])
        st.info("Prediction: Multiple Faces" if prediction[0] == 1 else "Prediction: Single/Two Faces")

        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        st.write("### Classification Report")
        st.json(report)

