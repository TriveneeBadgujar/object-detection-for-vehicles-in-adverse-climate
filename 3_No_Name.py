import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import threading
import os

# ---------------- VOICE (FINAL FIX) ----------------
last_spoken = ""

def speak_thread(msg):
    os.system(f'powershell -c "(New-Object -ComObject SAPI.SpVoice).Speak(\'{msg}\')"')

def speak_alert(message):
    global last_spoken

    if message == last_spoken:
        return

    last_spoken = message
    threading.Thread(target=speak_thread, args=(message,), daemon=True).start()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ADAS System", layout="wide")

# ---------------- STYLES ----------------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #00008B;
    text-align: center;
}

.weather-box {
    text-align: center;
    font-size: 26px;
    font-weight: 900;
    padding: 12px;
    border-radius: 12px;
    margin: 15px auto;
    width: 60%;
}

.clear { background-color: #14532d; color: #bbf7d0; }
.adverse { background-color: #7f1d1d; color: #fecaca; }

.card {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: bold;
}
.warning {background-color:#7f1d1d;color:#fecaca;}
.info {background-color:#1e3a8a;color:#bfdbfe;}
.safe {background-color:#14532d;color:#bbf7d0;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">Object Detection For Vehicles In Adverse Weather Condition</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    coco = YOLO("yolov8n.pt")
    custom = YOLO("best.pt")
    return coco, custom

coco_model, custom_model = load_models()

# ---------------- WEATHER FUNCTION ----------------
def detect_weather(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = gray.std()
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    # 🌫 Fog
    if contrast < 30:
        return "fog"

    # 🌧 Rain 
    elif edge_density > 25 and blur > 1800:
        return "rain"

    # ❄ Snow
    elif brightness > 200:
        return "snow"

    # ☀ Clear
    else:
        return "clear"

# ---------------- ADAS ----------------
ADAS_CLASSES = ["person","car","truck","bus","motorcycle","bicycle","traffic light","stop sign","dog","cow"]

def suggestions(labels, close):
    s = []
    voice_msgs = []

    if "person" in close:
        s.append("Brake! Pedestrian Ahead")
        voice_msgs.append("Warning! Pedestrian ahead")

    if "stop sign" in labels:
        s.append("Stop Sign Detected")
        voice_msgs.append("Stop sign ahead")

    if "traffic light" in labels:
        s.append("Traffic Signal Ahead")
        voice_msgs.append("Traffic signal ahead")

    if any(x in close for x in ["car","bus","truck"]):
        s.append("Vehicle Too Close")
        voice_msgs.append("Warning! Vehicle too close")

    if not s:
        s.append("Drive Safely")

    return s, voice_msgs

# ---------------- MODE ----------------
mode = st.sidebar.selectbox("Mode", ["Image","Video","Camera"])

# ================= IMAGE =================
if mode == "Image":
    file = st.file_uploader("Upload Image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        weather = detect_weather(img)

        if weather == "clear":
            active_model = coco_model
            box_class = "clear"
        else:
            active_model = custom_model
            box_class = "adverse"

        st.markdown(f'<div class="weather-box {box_class}">Weather: {weather.upper()}</div>', unsafe_allow_html=True)

        results = active_model(img)

        labels = set()
        close = set()

        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                name = active_model.names[cls]

                if name in ADAS_CLASSES:
                    labels.add(name)
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    area = (x2-x1)*(y2-y1)

                    color = (0,255,0)
                    if area > 60000:
                        close.add(name)
                        color = (0,0,255)
                        cv2.putText(img,"WARNING!",(30,40),0,1,(0,0,255),3)

                    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                    cv2.putText(img,name,(x1,y1-10),0,0.6,color,2)

        suggest_list, voice_msgs = suggestions(labels, close)

        col1,col2 = st.columns([3,2])

        with col1:
            st.image(img, channels="BGR")

        with col2:
            st.subheader("Suggestions")
            for x in suggest_list:
                if "Brake" in x:
                    st.markdown(f'<div class="card warning">{x}</div>', unsafe_allow_html=True)
                elif "Clear" in x:
                    st.markdown(f'<div class="card safe">{x}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card info">{x}</div>', unsafe_allow_html=True)

        #  VOICE
        for msg in voice_msgs:
            speak_alert(msg)

# ================= VIDEO =================
elif mode == "Video":
    file = st.file_uploader("Upload Video")

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            weather = detect_weather(frame)
            active_model = coco_model if weather=="clear" else custom_model

            results = active_model(frame)

            labels = set()
            close = set()

            for r in results:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    name = active_model.names[int(b.cls[0])]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    if name in ADAS_CLASSES:
                        labels.add(name)
                        area = (x2-x1)*(y2-y1)
                        if area > 60000:
                            close.add(name)

            suggest_list, voice_msgs = suggestions(labels, close)

            for msg in voice_msgs:
                speak_alert(msg)

            frame_window.image(frame, channels="BGR")

        cap.release()

# ================= CAMERA =================
elif mode == "Camera":
    run = st.checkbox("Start Camera")
    cap = cv2.VideoCapture(0)
    frame_window = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        weather = detect_weather(frame)
        active_model = coco_model if weather=="clear" else custom_model

        results = active_model(frame)

        labels = set()
        close = set()

        for r in results:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                name = active_model.names[int(b.cls[0])]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                if name in ADAS_CLASSES:
                    labels.add(name)
                    area = (x2-x1)*(y2-y1)
                    if area > 60000:
                        close.add(name)

        suggest_list, voice_msgs = suggestions(labels, close)

        for msg in voice_msgs:
            speak_alert(msg)

        frame_window.image(frame, channels="BGR")

    cap.release()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>WIET Student | Computer Engineering</center>", unsafe_allow_html=True)