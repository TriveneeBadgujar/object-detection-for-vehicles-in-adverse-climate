import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import threading
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ADAS System", layout="wide")

# ---------------- SIDEBAR STYLE ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dbeafe, #bfdbfe);
}

[data-testid="stSidebar"] * {
    color: #0f172a;
    font-family: 'Times New Roman', serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------- VOICE ----------------
last_spoken = ""

def speak_thread(msg):
    os.system(f'powershell -c "(New-Object -ComObject SAPI.SpVoice).Speak(\'{msg}\')"')

def speak_alert(message):
    global last_spoken
    if message == last_spoken:
        return
    last_spoken = message
    threading.Thread(target=speak_thread, args=(message,), daemon=True).start()

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Times New Roman', serif;
}

.stApp {
    background: linear-gradient(135deg, #e0f2fe, #f8fafc);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#4f46e5,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.weather-box {
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    padding: 12px 20px;
    border-radius: 10px;
    margin: 20px auto;
    width: fit-content;
}

.clear {
    background: linear-gradient(135deg, #bbf7d0, #86efac);
    color: #065f46;
}

.adverse {
    background: linear-gradient(135deg, #fecaca, #fca5a5);
    color: #7f1d1d;
}

.card {
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: 600;
}

.warning {background:#fecaca;color:#7f1d1d;}
.info {background:#dbeafe;color:#1e3a8a;}
.safe {background:#dcfce7;color:#14532d;}

.glass {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">ADAS System - Vehicle Detection</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), YOLO("best.pt")

coco_model, custom_model = load_models()

# ---------------- WEATHER ----------------
def detect_weather(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    if contrast < 30:
        return "fog"
    elif edge_density > 25 and blur > 1800:
        return "rain"
    elif brightness > 200:
        return "snow"
    else:
        return "clear"

# ---------------- ADAS ----------------
ADAS_CLASSES = ["person","car","truck","bus","motorcycle","bicycle","traffic light","stop sign","dog","cow"]

def suggestions(labels, close):
    s, voice = [], []

    if "person" in close:
        s.append("Brake! Pedestrian Ahead")
        voice.append("Warning pedestrian ahead")

    if "stop sign" in labels:
        s.append("Stop Sign Detected")
        voice.append("Stop sign ahead")

    if "traffic light" in labels:
        s.append("Traffic Signal Ahead")
        voice.append("Traffic signal ahead")

    if any(x in close for x in ["car","bus","truck"]):
        s.append("Vehicle Too Close")
        voice.append("Warning vehicle too close")

    if not s:
        s.append("Drive Safely")

    return s, voice

# ---------------- SIDEBAR ----------------
mode = st.sidebar.radio("Select Mode", ["Image","Video","Camera"])
sound_on = st.sidebar.checkbox("Enable Voice Alerts", value=True)

# ================= IMAGE =================
if mode == "Image":
    file = st.file_uploader("Upload Image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        model = coco_model if detect_weather(img) == "clear" else custom_model
        weather = detect_weather(img)

        st.markdown(f'<div class="weather-box {weather}">Weather: {weather.upper()}</div>', unsafe_allow_html=True)

        results = model(img)

        labels, close = set(), set()

        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                name = model.names[cls]

                if name in ADAS_CLASSES:
                    labels.add(name)
                    x1,y1,x2,y2 = map(int,b.xyxy[0])

                    area = (x2-x1)*(y2-y1)
                    color = (0,255,0)

                    if area > 60000:
                        close.add(name)
                        color = (0,0,255)

                    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                    cv2.putText(img,name,(x1,y1-10),0,0.6,color,2)

        alerts, voices = suggestions(labels, close)

        col1,col2 = st.columns([3,1])

        with col1:
            st.image(img, channels="BGR")

        with col2:
            st.subheader("Alerts")
            for a in alerts:
                st.markdown(f'<div class="card info">{a}</div>', unsafe_allow_html=True)

        if sound_on:
            for v in voices:
                speak_alert(v)

# ================= VIDEO =================
elif mode == "Video":
    file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        stop = st.button("Stop Video")

        while cap.isOpened():
            if stop:
                break

            ret, frame = cap.read()
            if not ret:
                break

            model = coco_model if detect_weather(frame) == "clear" else custom_model
            results = model(frame)

            for r in results:
                for b in r.boxes:
                    cls = int(b.cls[0])
                    name = model.names[cls]

                    if name in ADAS_CLASSES:
                        x1,y1,x2,y2 = map(int,b.xyxy[0])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,name,(x1,y1-10),0,0.6,(0,255,0),2)

            stframe.image(frame, channels="BGR")

# ================= CAMERA =================
elif mode == "Camera":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    stop = st.button("Stop Camera")

    while True:
        if stop:
            break

        ret, frame = cap.read()
        if not ret:
            break

        model = coco_model if detect_weather(frame) == "clear" else custom_model
        results = model(frame)

        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                name = model.names[cls]

                if name in ADAS_CLASSES:
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame,name,(x1,y1-10),0,0.6,(0,255,0),2)

        stframe.image(frame, channels="BGR")

# ================= TEAM SECTION (RESTORED) =================
st.markdown("""
<div class="glass">

<h3 style="text-align:center;">Project Team Members</h3>

<div style="text-align:center; margin-bottom:20px;">
<b>Prof. Rahul Jinturkar</b><br>
Project Guide - WIET Thane
</div>

<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:15px;">

<div class="card info">
<b>Trivenee Badgujar</b><br>Model Training
</div>

<div class="card info">
<b>Manjush Farad</b><br>Data Processing
</div>

<div class="card info">
<b>Aishwarya Khopade</b><br>Implementation
</div>

<div class="card info">
<b>Sonali Bhosle</b><br>Documentation
</div>

</div>
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>WIET Student | Computer Engineering</center>", unsafe_allow_html=True)