import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ======================
# Load Models
# ======================
@st.cache_resource
def load_models():
    seg_model = load_model("models/pgss_unet_best.h5", compile=False)   # <-- updated path
    cls_model = load_model("models/mera_net.h5", compile=False)         # <-- updated path
    return seg_model, cls_model

seg_model, cls_model = load_models()

# ======================
# Preprocessing Functions
# ======================
def preprocess_for_seg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (256, 256))
    norm = resized.astype("float32") / 255.0
    return np.expand_dims(norm, axis=(0, -1))

def preprocess_for_classification(img):
    resized = cv2.resize(img, (224, 224))
    norm = resized.astype("float32") / 255.0
    return np.expand_dims(norm, axis=0)

# ======================
# Segmentation Function
# ======================
def get_small_box_marked_image(img):
    H, W = img.shape[:2]
    inp = preprocess_for_seg(img)
    pred = seg_model.predict(inp)[0, :, :, 0]
    mask_resized = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    boxes_drawn = 0
    box_side = max(16, int(min(W, H) * 0.04))
    half_box = box_side // 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        cx = max(half_box, min(W - half_box - 1, cx))
        cy = max(half_box, min(H - half_box - 1, cy))
        x0, y0, x1, y1 = cx - half_box, cy - half_box, cx + half_box, cy + half_box
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.circle(output, (cx, cy), 3, (255, 0, 0), -1)
        boxes_drawn += 1

    return output, boxes_drawn

# ======================
# Classification Function
# ======================
def classify_image(img):
    inp = preprocess_for_classification(img)
    preds = cls_model.predict(inp)[0]
    classes = ["Malignant", "Benign", "Normal"]
    idx = np.argmax(preds)
    return classes[idx], preds, classes

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Breast Cancer Analyzer", layout="wide")

st.title("🩺 Breast Cancer Analyzer")
st.write("Upload an image and select a task from the sidebar.")

uploaded_file = st.file_uploader("Upload a mammogram image", type=["png", "jpg", "jpeg"])

# ---- Sidebar with Radio Button ----
st.sidebar.title("Choose Task")

# Using colored labels with emojis
task = st.sidebar.radio(
    "Select Task:",
    options=[
        "🟦 Segmentation",
        "🟩 Classification",
        "🟪 Prediction"
    ]
)

# ---- Sidebar background color ----
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #f0f4f8;
}
</style>
""", unsafe_allow_html=True)

# ---- Image Display & Processing ----
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Input Image", use_container_width=True)

    if task.startswith("🟦"):  # Segmentation
        with st.spinner("Detecting suspicious regions..."):
            out_img, nboxes = get_small_box_marked_image(img)
        if nboxes > 0:
            col2.image(out_img, caption="Suspicious Regions Marked", use_container_width=True)
        else:
            col2.warning("No suspicious regions detected.")

    elif task.startswith("🟩"):  # Classification
        with st.spinner("Classifying image..."):
            label, preds, classes = classify_image(img)
        col2.success(f"Prediction: **{label}**")
        conf = preds[np.argmax(preds)] * 100
        col2.write(f"Confidence: {conf:.2f}%")

    elif task.startswith("🟪"):  # Prediction
        with st.spinner("Generating prediction scores..."):
            label, preds, classes = classify_image(img)
        col2.subheader("Prediction Probabilities:")
        for cls, prob in zip(classes, preds):
            col2.write(f"**{cls}**: {prob * 100:.2f}%")
        col2.success(f"Final Decision: **{label}**")
