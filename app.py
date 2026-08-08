"""
Cigarette Detection App — YOLOv8
Author: Nadeem Gohar
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2
import os
import time

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cigarette Detection | Nadeem Gohar",
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark / glassmorphism styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e6e6e6;
    }
    section[data-testid="stSidebar"] {
        background: rgba(20, 20, 35, 0.85);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: rgba(255, 87, 34, 0.12);
        border: 1px solid rgba(255, 87, 34, 0.35);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        margin-top: 3rem;
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(255,255,255,0.25);
        border-radius: 12px;
        padding: 1rem;
    }
    .danger-banner {
        background: rgba(220, 38, 38, 0.18);
        border: 1px solid rgba(220, 38, 38, 0.6);
        border-left: 5px solid #dc2626;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.75rem 0;
        color: #ffb4b4;
        font-weight: 600;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = {0: "cigarette"}  # must match training (single class)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = "model/best.pt"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown(
        "Real-time cigarette detection powered by a **YOLOv8** model, "
        "trained on a custom-annotated dataset. Upload an image or video "
        "to flag smoking with bounding boxes and confidence scores."
    )
    st.markdown("---")
    st.markdown("Built by **Nadeem Gohar**")
    st.markdown("[GitHub](https://github.com/Nadeemgohar) · [LinkedIn](https://linkedin.com/in/nadeem-gohar-0708382b0/)")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🚬 Cigarette Detection")
st.markdown("##### High-efficiency cigarette detection using YOLOv8 — flags smoking in restricted/public areas from images or video.")
st.markdown("---")

if model is None:
    st.error(
        f"⚠️ Model weights not found at `{MODEL_PATH}`. "
        "Train the model using the Colab notebook, download `best.pt`, "
        "and place it at `model/best.pt` in this project folder."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs: Image / Video
# ---------------------------------------------------------------------------
tab_img, tab_vid = st.tabs(["🖼️ Image Detection", "🎬 Video Detection"])

# --- Image tab -------------------------------------------------------------
with tab_img:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img_upload")

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            start = time.time()
            results = model.predict(np.array(image), conf=conf_threshold, iou=iou_threshold, verbose=False)
            elapsed = time.time() - start

        annotated = results[0].plot()[:, :, ::-1]  # BGR -> RGB

        with col2:
            st.markdown("**Detected**")
            st.image(annotated, use_container_width=True)

        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0

        cig_count = 0
        if num_detections > 0:
            classes = boxes.cls.tolist()
            cig_count = sum(1 for c in classes if CLASS_NAMES.get(int(c)) == "cigarette")

        if cig_count > 0:
            st.markdown(
                f'<div class="danger-banner">⚠️ SMOKING VIOLATION DETECTED — '
                f'{cig_count} cigarette{"s" if cig_count != 1 else ""} identified. '
                f'This may be a restricted/no-smoking area.</div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-box"><h3>{num_detections}</h3>Cigarettes found</div>', unsafe_allow_html=True)
        with m2:
            avg_conf = float(boxes.conf.mean()) if num_detections > 0 else 0.0
            st.markdown(f'<div class="metric-box"><h3>{avg_conf:.1%}</h3>Avg. confidence</div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><h3>{elapsed*1000:.0f} ms</h3>Inference time</div>', unsafe_allow_html=True)

        if num_detections == 0:
            st.info("No cigarettes detected above the confidence threshold. Try lowering it in the sidebar.")

# --- Video tab ---------------------------------------------------------------
with tab_vid:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="vid_upload")

    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())

        st.markdown("**Processing video (frame-by-frame)** — this may take a moment for longer clips.")
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        total_detections = 0
        total_cig_detections = 0
        cig_alert_placeholder = st.empty()

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated_frame = results[0].plot()
            frame_boxes = results[0].boxes
            frame_count = len(frame_boxes) if frame_boxes is not None else 0
            total_detections += frame_count

            frame_cig_count = 0
            if frame_count > 0:
                frame_cig_count = sum(1 for c in frame_boxes.cls.tolist() if CLASS_NAMES.get(int(c)) == "cigarette")
                total_cig_detections += frame_cig_count

            writer.write(annotated_frame)

            if frame_idx % 5 == 0:
                frame_placeholder.image(annotated_frame[:, :, ::-1], use_container_width=True)
                if frame_cig_count > 0:
                    cig_alert_placeholder.markdown(
                        '<div class="danger-banner">⚠️ SMOKING VIOLATION DETECTED in current frame — '
                        'cigarette identified.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    cig_alert_placeholder.empty()
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()

        st.success(
            f"Done — processed {frame_idx} frames, {total_detections} total detections "
            f"({total_cig_detections} cigarette instances across all frames)."
        )
        if total_cig_detections > 0:
            st.markdown(
                '<div class="danger-banner">⚠️ This video contains one or more smoking violations.</div>',
                unsafe_allow_html=True,
            )
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download annotated video", f, file_name="cigarette_detection_output.mp4")

        os.unlink(tfile.name)

# ---------------------------------------------------------------------------
st.markdown('<div class="footer">Cigarette Detection App · YOLOv8 · Built by Nadeem Gohar</div>', unsafe_allow_html=True)
