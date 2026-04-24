import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np
import threading
import os

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

# ---------------- PREMIUM LIGHT UI ----------------
st.markdown("""
<style>       

/* Global Font */
html, body, [class*="css"]  {
    font-family: 'Times New Roman', serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #e0f2fe, #f8fafc);
}

            
/* Title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg,#4f46e5,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Weather */
.weather-box {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    padding: 12px;
    border-radius: 12px;
    margin: 15px auto;
    width: 50%;
}

.clear { background-color: #d1fae5; color: #065f46; }
.adverse { background-color: #fee2e2; color: #991b1b; }

/* Cards */
.card {
    padding: 14px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: 600;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.03);
}

.warning {background:#fecaca;color:#7f1d1d;}
.info {background:#dbeafe;color:#1e3a8a;}
.safe {background:#dcfce7;color:#14532d;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a, #2563eb);
    color: white;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: white !important;
}

/* Sidebar title */
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
}
            
# Team Member 
.team-section {
    margin-top: 20px;
}

/* Professor Box */
.professor {
    text-align: center;
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    border: 1px solid #93c5fd;
    margin-bottom: 20px;
}

/* Grid */
.team-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
}

/* Cards */
.team-card {
    padding: 15px;
    border-radius: 12px;
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(0,0,0,0.05);
    text-align: center;
    transition: 0.3s;
}

.team-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

/* Role styling */
.role {
    color: #2563eb;
    font-weight: 600;
    font-size: 14px;
}

/* Email */
.team-card a {
    text-decoration: none;
    color: #dc2626;
    font-weight: 500;
}

.team-card a:hover {
    text-decoration: underline;
}

            .mail-btn {
    display: inline-block;
    margin-top: 8px;
    padding: 6px 12px;
    background: #2563eb;
    color: white !important;
    border-radius: 6px;
    font-size: 13px;
    text-decoration: none;
}

.mail-btn:hover {
    background: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)




# ---------------- TITLE ----------------
st.markdown('<div class="main-title">Object Detection For Vehicles In Adverse Weather Condition</div>', unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    coco = YOLO("yolov8n.pt")
    custom = YOLO("best.pt")
    return coco, custom

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
        return "fog"

# ---------------- ADAS ----------------
ADAS_CLASSES = ["person","car","truck","bus","motorcycle","bicycle","traffic light","stop sign","cow"]

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
    
    if any(x in close for x in ["dog","cow"]):
        s.append("Animal Too Close")
        voice_msgs.append("Warning! Animal too close")

    if not s:
        s.append("Drive Safely")

    return s, voice_msgs

# ---------------- SIDEBAR ----------------
st.sidebar.markdown('<div class="sidebar-title">System Controls</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("Select Mode", ["Image","Video","Camera"])
sound_on = st.sidebar.checkbox("Enable Voice Alerts", value=True)


def resize_image(img, width=1400, height=700):
    return cv2.resize(img, (width, height))

# ================= IMAGE =================
if mode == "Image":
    file = st.file_uploader("Upload Image")

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

        weather = detect_weather(img)
        active_model = coco_model if weather=="clear" else custom_model
        box_class = "clear" if weather=="clear" else "adverse"

        st.markdown(f'<div class="weather-box {box_class}">Weather: {weather.upper()}</div>', unsafe_allow_html=True)

        with st.spinner("Processing..."):
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
                        cv2.putText(img,"WARNING",(30,40),0,1,(0,0,255),3)

                    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                    cv2.putText(img,name,(x1,y1-10),0,0.6,color,2)

        suggest_list, voice_msgs = suggestions(labels, close)

        col1,col2 = st.columns([3,1])
        
        with col1:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.image(img, channels="BGR", use_container_width=True)
            # st.image(img, channels="BGR")
            # resized_img = resize_image(img)
            #st.image(resized_img, channels="BGR")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("System Alerts")

            for x in suggest_list:
                if "Brake" in x:
                    st.markdown(f'<div class="card warning">{x}</div>', unsafe_allow_html=True)
                elif "Safely" in x:
                    st.markdown(f'<div class="card safe">{x}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card info">{x}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Stats Panel (UPDATED)
    
        if sound_on:
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

                    if name in ADAS_CLASSES:
                        labels.add(name)

                        area = (x2-x1)*(y2-y1)
                        color = (0,255,0)

                        if area > 60000:
                            close.add(name)
                            color = (0,0,255)
                            cv2.putText(frame,"WARNING",(30,40),0,1,(0,0,255),3)

                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        cv2.putText(frame, name, (x1,y1-10), 0, 0.6, color, 2)

            suggest_list, voice_msgs = suggestions(labels, close)

            if sound_on:
                for msg in voice_msgs:
                    speak_alert(msg)

            # Fit to screen (BEST UI)
            # frame_window.image(frame, channels="BGR", use_container_width=True)
            resized_frame = resize_image(frame) 
            frame_window.image(resized_frame, channels="BGR")

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

                if name in ADAS_CLASSES:
                    labels.add(name)

                    area = (x2-x1)*(y2-y1)
                    color = (0,255,0)

                    if area > 60000:
                        close.add(name)
                        color = (0,0,255)
                        cv2.putText(frame,"WARNING",(30,40),0,1,(0,0,255),3)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame, name, (x1,y1-10), 0, 0.6, color, 2)

        suggest_list, voice_msgs = suggestions(labels, close)

        if sound_on:
            for msg in voice_msgs:
                speak_alert(msg)

        # Fit to screen
        # frame_window.image(frame, channels="BGR", use_container_width=True)
        resized_frame = resize_image(frame) 
        frame_window.image(resized_frame, channels="BGR")

    cap.release()

st.markdown("""
<div class="glass team-section">

<h3 style="text-align:center;">Project Team Member</h3>

<!-- Professor -->
<div class="team-card professor">
<b>Prof. Rahul Jinturkar</b><br>
<span class="role">Project Guide</span><br>
Computer Engineering<br>
Watumull Institute of Engineering and Technology<br>
Thane, India<br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=rjinturkar@yahoo.com" target="_blank">
rjinturkar@yahoo.com
</a><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=rjinturkar@yahoo.com" target="_blank" class="mail-btn">
✉ Send Mail
</a>
</div>

<!-- Students Grid -->
<div class="team-grid">

<div class="team-card">
<b>Trivenee Badgujar</b><br>
<span class="role">Model Training & GUI Development</span><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=triveneebadgujar28@gmail.com" target="_blank">
triveneebadgujar28@gmail.com
</a><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=triveneebadgujar28@gmail.com" target="_blank" class="mail-btn">
✉ Send Mail
</a>
</div>
            
<div class="team-card">
<b>Aishwarya Khopade</b><br>
<span class="role">Project Support & Implementation</span><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=aishwaryakhopade686@gmail.com" target="_blank">
aishwaryakhopade686@gmail.com
</a><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=aishwaryakhopade686@gmail.com" target="_blank" class="mail-btn">
✉ Send Mail
</a>
</div>            

<div class="team-card">
<b>Manjush Farad</b><br>
<span class="role">Data Collection & Processing</span><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=manjushfarad@gmail.com" target="_blank">
manjushfarad@gmail.com
</a><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=manjushfarad@gmail.com" target="_blank" class="mail-btn">
✉ Send Mail
</a>
</div>


<div class="team-card">
<b>Sonali Bhosle</b><br>
<span class="role">Project Support & Documentation</span><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=sonalibhosle605@gmail.com" target="_blank">
sonalibhosle605@gmail.com
</a><br>

<a href="https://mail.google.com/mail/?view=cm&fs=1&to=sonalibhosle605@gmail.com" target="_blank" class="mail-btn">
✉ Send Mail
</a>
</div>

</div>

</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>WIET Student | Computer Engineering</center>", unsafe_allow_html=True)
