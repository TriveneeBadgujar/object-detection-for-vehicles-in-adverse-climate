import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import threading
import os
from collections import Counter

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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ADAS System", layout="wide")

# ---------------- STYLES ----------------
st.markdown("""
<style>
.main-title {font-size:38px;font-weight:800;color:#00008B;text-align:center;}
.weather-box {text-align:center;font-size:26px;font-weight:900;padding:12px;border-radius:12px;margin:15px auto;width:60%;}
.clear {background-color:#14532d;color:#bbf7d0;}
.adverse {background-color:#7f1d1d;color:#fecaca;}
.card {padding:12px;border-radius:10px;margin-bottom:10px;font-weight:bold;}
.warning {background-color:#7f1d1d;color:#fecaca;}
.info {background-color:#1e3a8a;color:#bfdbfe;}
.safe {background-color:#14532d;color:#bbf7d0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Object Detection For Vehicles In Adverse Weather Condition</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    return YOLO("yolov8n.pt"), YOLO("best.pt")

coco_model, custom_model = load_models()

# ---------------- WEATHER ----------------
weather_history = []
current_weather = None
current_model = coco_model

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

def get_stable_weather(frame):
    global weather_history

    w = detect_weather(frame)
    weather_history.append(w)

    if len(weather_history) > 5:
        weather_history.pop(0)

    return Counter(weather_history).most_common(1)[0][0]

def get_model(weather):
    global current_weather, current_model

    if weather != current_weather:
        current_weather = weather
        if weather == "clear":
            current_model = coco_model
        else:
            current_model = custom_model

    return current_model

# ---------------- ADAS ----------------
ADAS_CLASSES = [
    # Humans
    "person",

    # Vehicles
    "car", "truck", "bus", "motorcycle", "bicycle",

    # Traffic control
    "traffic light", "stop sign",

    # Road obstacles / animals
    "dog", "cow", "horse", "sheep", "cat",

    # Road-side objects (important for context)
    "bench", "parking meter",

    # Optional (useful in some cases)
    "fire hydrant"
]

def suggestions(labels, close):
    s, voice_msgs = [], []

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
        active_model = coco_model if weather=="clear" else custom_model

        st.markdown(f'<div class="weather-box {"clear" if weather=="clear" else "adverse"}">Weather: {weather.upper()}</div>', unsafe_allow_html=True)
        st.write(f"Model Used: {'COCO' if active_model==coco_model else 'CUSTOM'}")  # DEBUG

        results = active_model(img)

        labels, close = set(), set()

        for r in results:
            for b in r.boxes:
                name = active_model.names[int(b.cls[0])]
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
                st.markdown(f'<div class="card {"warning" if "Brake" in x else "info"}">{x}</div>', unsafe_allow_html=True)

        for msg in voice_msgs:
            speak_alert(msg)

# ================= VIDEO & CAMERA (FIXED) =================
elif mode in ["Video","Camera"]:

    cap = None

    if mode == "Video":
        file = st.file_uploader("Upload Video")
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)

    else:
        run = st.checkbox("Start Camera")
        if run:
            cap = cv2.VideoCapture(0)

    frame_window = st.empty()

    if cap:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            weather = get_stable_weather(frame)   # 🔥 SMOOTH
            active_model = get_model(weather)     # 🔥 LOCK

            results = active_model(frame)

            labels, close = set(), set()

            for r in results:
                for b in r.boxes:
                    name = active_model.names[int(b.cls[0])]
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    if name in ADAS_CLASSES:
                        labels.add(name)
                        if (x2-x1)*(y2-y1) > 60000:
                            close.add(name)

            suggest_list, voice_msgs = suggestions(labels, close)

            for msg in voice_msgs:
                speak_alert(msg)

            frame_window.image(frame, channels="BGR")

        cap.release()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>WIET Student | Computer Engineering</center>", unsafe_allow_html=True)