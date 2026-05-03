import os
import csv
import numpy as np
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models



# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)
with st.sidebar:
    st.title("🫁 Pneumonia AI")
    st.write("Chest X-ray based pneumonia detection system")

    st.markdown("---")
    st.subheader("Model")
    st.write("Architecture: ResNet18")
    st.write("Classes: Normal / Pneumonia")
    st.write("Input Size: 224 × 224")

    st.markdown("---")
    st.subheader("How to use")
    st.write("1. Enter patient details")
    st.write("2. Upload chest X-ray")
    st.write("3. View prediction")
    st.write("4. Download report")

# ----------------------------
# Constants
# ----------------------------
MODEL_PATH = "model/pneumonia_model.pth"
HISTORY_PATH = "prediction_history.csv"
CLASSES = ["Normal", "Pneumonia"]


# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(MODEL_PATH):
        return None

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


model = load_model()


# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
def is_likely_xray(image):
    """
    Basic validation to reject screenshots, forms, colorful images, etc.
    This does not medically verify the image, but helps avoid wrong predictions
    on non-X-ray inputs.
    """

    img = image.convert("RGB")
    arr = np.array(img)

    height, width, _ = arr.shape
    aspect_ratio = width / height

    # Chest X-rays are usually not extremely wide like screenshots
    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
        return False

    # X-rays are mostly grayscale, so RGB channels should be similar
    r = arr[:, :, 0].astype("float32")
    g = arr[:, :, 1].astype("float32")
    b = arr[:, :, 2].astype("float32")

    color_difference = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))

    if color_difference > 35:
        return False

    # Reject images that are too bright or too dark overall
    gray = np.mean(arr, axis=2)
    mean_brightness = np.mean(gray)

    if mean_brightness < 15 or mean_brightness > 240:
        return False

    return True

# ----------------------------
# Helper Functions
# ----------------------------
def predict_image(image):
    image = image.convert("RGB")
    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    prediction = CLASSES[pred.item()]
    confidence_score = confidence.item() * 100

    return prediction, confidence_score


def save_prediction(patient_name, age, gender, symptoms, filename, prediction, confidence):
    file_exists = os.path.exists(HISTORY_PATH)

    with open(HISTORY_PATH, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date Time",
                "Patient Name",
                "Age",
                "Gender",
                "Symptoms",
                "File Name",
                "Prediction",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            patient_name,
            age,
            gender,
            symptoms,
            filename,
            prediction,
            f"{confidence:.2f}%"
        ])
    


def generate_report(patient_name, age, gender, symptoms, filename, prediction, confidence):
    report = f"""
PNEUMONIA DETECTION REPORT

Date & Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Patient Details:
Patient Name: {patient_name}
Age: {age}
Gender: {gender}
Symptoms: {symptoms}

Image Details:
Uploaded File: {filename}

Prediction Result:
Prediction: {prediction}
Confidence: {confidence:.2f}%

Medical Disclaimer:
This tool is created for educational and demonstration purposes only.
It should not be used as a replacement for professional medical diagnosis.
Please consult a qualified doctor or radiologist for final medical decision.
"""
    return report


# ----------------------------
# UI
# ----------------------------
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload chest X-ray images and get prediction using a ResNet18 deep learning model.")

# st.warning(
#     "⚠️ Disclaimer: This project is for educational/demo purposes only. "
#     "It should not replace professional medical diagnosis."
# )

if model is None:
    st.error("Model file not found: model/pneumonia_model.pth")
    st.info(
        "Please train the model first using:\n\n"
        "1. cd src\n"
        "2. python train.py\n\n"
        "After training, run the app again."
    )
    st.stop()


# ----------------------------
# Patient Form
# ----------------------------
st.subheader("👤 Patient Information")

patient_name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
symptoms = st.text_area("Symptoms / Notes", placeholder="Example: cough, fever, chest pain")


# ----------------------------
# Upload Section
# ----------------------------
# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📤 Upload X-ray Image")

uploaded_files = st.file_uploader(
    "Upload one or more X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()

        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption=f"Uploaded Image: {uploaded_file.name}",
            use_container_width=True
        )

        if not is_likely_xray(image):
            st.error("Invalid input image. Please upload a proper chest X-ray image.")
            st.warning(
                "This model is trained only on chest X-ray images, "
                "not screenshots, forms, selfies, or normal photos."
            )
            continue

        with st.spinner("Analyzing X-ray image..."):
            prediction, confidence = predict_image(image)

        st.subheader("Prediction Result")

        if prediction == "Pneumonia":
            st.error(f"Result: {prediction}")
            st.error("⚠️ Risk Level: High")
        else:
            st.success(f"Result: {prediction}")
            st.success("✅ Risk Level: Low")

        st.info(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))

        if confidence < 75:
            st.warning(
                "Low confidence prediction. Please consult a medical professional "
                "and do not rely only on this result."
            )

        save_prediction(
            patient_name,
            age,
            gender,
            symptoms,
            uploaded_file.name,
            prediction,
            confidence
        )

        report = generate_report(
            patient_name,
            age,
            gender,
            symptoms,
            uploaded_file.name,
            prediction,
            confidence
        )

        st.download_button(
            label="📄 Download Prediction Report",
            data=report,
            file_name=f"report_{uploaded_file.name}.txt",
            mime="text/plain"
        )

# ----------------------------
# History Section
# ----------------------------
st.divider()
st.subheader("📊 Prediction History")

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history_data = file.read()

    st.download_button(
        label="⬇️ Download Prediction History CSV",
        data=history_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.checkbox("Show prediction history"):
        st.text(history_data)
else:
    st.write("No predictions saved yet.")