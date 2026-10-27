import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
TRAIN_DIR = BASE_DIR / "dataset" / "train"

# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f3faf4;
    color: #1b4332;
}

.stApp p,
.stApp label,
.stApp li,
.stApp span,
.stApp div {
    color: #1b4332;
}

h1, h2, h3, h4 {
    color: #166534 !important;
}

.main-title {
    color: #166534 !important;
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 8px;
}

.subtitle {
    color: #374151 !important;
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

.result-card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #bbf7d0;
    margin: 12px 0;
}

.result-heading {
    color: #166534 !important;
    font-size: 23px;
    font-weight: bold;
}

.result-disease {
    color: #14532d !important;
    font-size: 27px;
    font-weight: bold;
    overflow-wrap: anywhere;
}

.result-confidence {
    color: #047857 !important;
    font-size: 32px;
    font-weight: 800;
}

.info-card {
    background-color: #ffffff;
    border-left: 5px solid #16a34a;
    border-radius: 10px;
    padding: 16px;
    margin: 12px 0;
}

.info-card h3 {
    color: #166534 !important;
}

.info-card p,
.info-card li {
    color: #1f2937 !important;
}

.stButton button {
    background-color: #15803d;
    color: white !important;
    border-radius: 9px;
    border: none;
    font-weight: bold;
}

.stButton button p {
    color: white !important;
}

[data-testid="stMetric"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #1b4332 !important;
}

[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] * {
    color: #1b4332 !important;
}

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] * {
    color: #1b4332 !important;
}

.stCaption,
.stCaption * {
    color: #374151 !important;
}

hr {
    border-color: #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🌱 Plant Disease Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered plant leaf analysis using MobileNetV2 and Transfer Learning'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload a clear image of a plant leaf to predict its class "
    "and view the model confidence."
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# ---------------- LOAD CLASS NAMES ----------------

def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            names = json.load(file)

        if isinstance(names, list) and names:
            return names

    if TRAIN_DIR.exists():
        names = sorted(
            folder.name
            for folder in TRAIN_DIR.iterdir()
            if folder.is_dir()
        )

        if names:
            return names

    return None


class_names = load_class_names()

if not class_names:
    st.error(
        "Class names could not be loaded. Please check "
        "models/class_names.json or the dataset/train folders."
    )
    st.stop()

# ---------------- IMAGE UPLOAD ----------------

st.markdown("## 📷 Upload a Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Uploaded Image")
            st.image(
                image,
                caption="Uploaded plant leaf",
                use_container_width=True
            )

        with right:
            st.markdown("### Image Details")
            st.write(f"**Image size:** {image.width} × {image.height}")
            st.write("**Image format:** RGB")
            st.write(f"**Model classes:** {len(class_names)}")

        if st.button("🔍 Detect Plant Disease", use_container_width=True):

            with st.spinner("Analyzing the leaf image..."):

                # The model includes the same rescaling layer
                # used during training. Do not normalize here again.
                resized_image = image.resize((224, 224))

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "The number of model outputs does not match "
                        "the number of class names. Check your training "
                        "class order and class_names.json."
                    )
                    st.stop()

                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

                st.markdown("---")
                st.markdown("## 🌿 Prediction Results")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-heading">Predicted Class</div>
                        <div class="result-disease">
                            {predicted_class}
                        </div>
                        <br>
                        <div class="result-heading">
                            Model Confidence
                        </div>
                        <div class="result-confidence">
                            {confidence:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(confidence / 100, 0.0), 1.0)
                )

                st.caption(
                    "Confidence is the model's score for its top prediction; "
                    "it is not a guarantee that the prediction is correct."
                )

                # ---------------- TOP 3 PREDICTIONS ----------------

                st.markdown("### 📊 Top Predictions")

                top_indices = np.argsort(predictions)[::-1][
                    :min(3, len(predictions))
                ]

                for index in top_indices:
                    score = float(predictions[index]) * 100

                    st.write(
                        f"**{class_names[index]}** — {score:.2f}%"
                    )

                    st.progress(
                        min(max(score / 100, 0.0), 1.0)
                    )

                # ---------------- EXPLAINABLE AI NOTE ----------------

                st.markdown("### 🧠 Understanding the Result")

                st.markdown(
                    """
                    <div class="info-card">
                        <p>
                        The system uses MobileNetV2, a convolutional neural
                        network, to classify the uploaded leaf image.
                        The confidence percentage represents the model's
                        score for the predicted class.
                        </p>
                        <p>
                        This score alone does not explain which parts of
                        the leaf influenced the prediction. A Grad-CAM
                        heatmap can be added separately for visual
                        explainability.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---------------- DISEASE INFORMATION ----------------

                st.markdown("## 🌱 Disease Information & Precautions")

                class_lower = predicted_class.lower()

                if "healthy" in class_lower:
                    st.success(
                        "The model classified this leaf as healthy. "
                        "Continue regular watering and monitor the plant "
                        "for changes."
                    )

                elif any(
                    word in class_lower
                    for word in [
                        "bacterial", "blight", "spot",
                        "mold", "mildew", "rot",
                        "rust", "virus", "scab"
                    ]
                ):
                    st.warning(
                        "The model detected a class associated with "
                        "plant disease. This is an AI prediction, not a "
                        "confirmed diagnosis."
                    )

                else:
                    st.info(
                        "Review the predicted class and compare the leaf "
                        "with reliable plant-health guidance."
                    )

                st.markdown(
                    """
                    <div class="info-card">
                        <h3>General Plant Care</h3>
                        <ul>
                            <li>Inspect leaves regularly for unusual spots or changes.</li>
                            <li>Avoid overwatering and ensure proper drainage.</li>
                            <li>Provide suitable sunlight and air circulation.</li>
                            <li>Keep potentially infected plants separate while investigating.</li>
                            <li>Confirm suspected diseases with a local agricultural expert
                            before applying treatments.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as error:
        st.error(f"Could not process the uploaded image: {error}")

else:
    st.markdown(
        """
        <div class="info-card">
            <h3>How to Use</h3>
            <ol>
                <li>Upload a clear plant leaf image.</li>
                <li>Click Detect Plant Disease.</li>
                <li>Review the predicted class and confidence percentage.</li>
                <li>Compare the prediction with the actual leaf condition.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption(
    "Academic project: Explainable AI-Based Plant Disease Detection "
    "Using MobileNetV2 and Transfer Learning."
)
