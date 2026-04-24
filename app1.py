import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ADAS System", layout="wide")

# ---------------- CUSTOM UI STYLES ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #00008B;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #0041C2;
    font-size: 28px;
}

.sidebar-title {
    font-size: 28px;
    font-weight: bold;
    color: #38bdf8;
}

.card {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    font-size: 17px;
    font-weight: 600;
}

.warning {
    background-color: #7f1d1d;
    color: #fecaca;
}

.info {
    background-color: #1e3a8a;
    color: #bfdbfe;
}

.safe {
    background-color: #14532d;
    color: #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">Vision Based Object Detection for Vehicles in Adverse Weather Condition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Object Detection & Driver Assistance using YOLOv8</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)

# ✅ MODEL SELECTION (ADDED)
model_choice = st.sidebar.radio(
    "🧠 Select Detection Model",
    ("Pretrained COCO (YOLOv8)", "Custom BDD100K (best.pt)")
)

# ✅ LOAD MODEL (ADDED)
@st.cache_resource
def load_model(choice):
    if choice == "Pretrained COCO (YOLOv8)":
        return YOLO("yolov8n.pt")
    else:
        return YOLO("best.pt")

model = load_model(model_choice)

# ---------------- ADAS CLASSES ----------------
ADAS_CLASSES = [
    "person", "car", "truck", "bus", "motorcycle",
    "bicycle", "train", "traffic light",
    "stop sign", "dog", "cat", "cow"
]

# ---------------- DRIVER ASSISTANCE LOGIC ----------------
def get_driver_suggestions(detected_labels, close_objects):
    suggestions = []

    if "traffic light" in detected_labels:
        suggestions.append("Traffic Signal Ahead – Be Ready to Stop")

    if "stop sign" in detected_labels:
        suggestions.append("Stop Sign Ahead – Slow Down")

    if "person" in close_objects:
        suggestions.append("Pedestrian Ahead – Brake Immediately")

    if any(obj in close_objects for obj in ["car", "bus", "truck", "motorcycle"]):
        suggestions.append("Vehicle Too Close – Maintain Safe Distance")

    if any(obj in detected_labels for obj in ["dog", "cat", "cow"]):
        suggestions.append("Animal on Road – Drive Carefully")

    if not suggestions:
        suggestions.append("Road Clear – Drive Safely")

    return suggestions

# ---------------- MODE SELECTION ----------------
mode = st.sidebar.selectbox(
    "Select Input Mode",
    ("Upload Image", "Upload Video", "Live Camera")
)

st.sidebar.markdown("""
### Features
- 🚦 Traffic Signal Detection  
- 🚶 Pedestrian Warning  
- 🚗 Collision Alert  
- 🐾 Animal Detection  
""")

# ================= IMAGE MODE =================
if mode == "Upload Image":
    uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        detected_labels = set()
        close_objects = set()

        results = model(img)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label in ADAS_CLASSES:
                    detected_labels.add(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    color = (0, 255, 0)
                    if area > 60000:
                        close_objects.add(label)
                        color = (0, 0, 255)
                        cv2.putText(img, "WARNING: CLOSE OBJECT!",
                                    (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 3)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        suggestions = get_driver_suggestions(detected_labels, close_objects)

        col1, col2 = st.columns([3, 2])

        with col1:
            st.image(img, channels="BGR", caption="Detection Output")

        with col2:
            st.markdown("### Driver Assistance Suggestions")
            for s in suggestions:
                if "Brake" in s or "Stop" in s or "Too Close" in s:
                    st.markdown(f'<div class="card warning">{s}</div>', unsafe_allow_html=True)
                elif "Clear" in s:
                    st.markdown(f'<div class="card safe">{s}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card info">{s}</div>', unsafe_allow_html=True)

# ================= VIDEO MODE =================
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()
        suggestion_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detected_labels = set()
            close_objects = set()

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    if label in ADAS_CLASSES:
                        detected_labels.add(label)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)

                        color = (0, 255, 0)
                        if area > 60000:
                            close_objects.add(label)
                            color = (0, 0, 255)
                            cv2.putText(frame, "WARNING: CLOSE OBJECT!",
                                        (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stframe.image(frame, channels="BGR")

            suggestions = get_driver_suggestions(detected_labels, close_objects)
            suggestion_box.markdown("### Driver Assistance Suggestions")

            for s in suggestions:
                if "Brake" in s or "Stop" in s:
                    suggestion_box.markdown(f'<div class="card warning">{s}</div>', unsafe_allow_html=True)
                elif "Clear" in s:
                    suggestion_box.markdown(f'<div class="card safe">{s}</div>', unsafe_allow_html=True)
                else:
                    suggestion_box.markdown(f'<div class="card info">{s}</div>', unsafe_allow_html=True)

        cap.release()

# ================= LIVE CAMERA MODE =================
elif mode == "Live Camera":
    run = st.checkbox("▶️ Start Camera")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    suggestion_box = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        detected_labels = set()
        close_objects = set()

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label in ADAS_CLASSES:
                    detected_labels.add(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    color = (0, 255, 0)
                    if area > 60000:
                        close_objects.add(label)
                        color = (0, 0, 255)
                        cv2.putText(frame, "WARNING: CLOSE OBJECT!",
                                    (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 3)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stframe.image(frame, channels="BGR")

        suggestions = get_driver_suggestions(detected_labels, close_objects)
        suggestion_box.markdown("### Driver Assistance Suggestions")

        for s in suggestions:
            if "Brake" in s or "Stop" in s:
                suggestion_box.markdown(f'<div class="card warning">{s}</div>', unsafe_allow_html=True)
            elif "Clear" in s:
                suggestion_box.markdown(f'<div class="card safe">{s}</div>', unsafe_allow_html=True)
            else:
                suggestion_box.markdown(f'<div class="card info">{s}</div>', unsafe_allow_html=True)

    cap.release()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center> WIET Student | Computer Engineering</center>", unsafe_allow_html=True)
