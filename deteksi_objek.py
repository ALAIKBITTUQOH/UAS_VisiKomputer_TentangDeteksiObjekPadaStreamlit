import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model ringan
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="Deteksi Objek Gambar & Video", layout="centered")
st.title("🚀 Deteksi Objek pada Gambar dan Video dengan YOLO")

menu = st.sidebar.radio("Pilih Mode", ["Deteksi Gambar", "Deteksi Video"])

# Fungsi untuk deteksi gambar
def detect_image(image):
    results = model.predict(image)
    result_image = results[0].plot()
    boxes = results[0].boxes
    labels = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        labels.append(label)
    return result_image, labels

# Fungsi untuk deteksi video (minim delay)
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame agar lebih ringan
        frame = cv2.resize(frame, (640, 360))

        # Prediksi
        results = model.predict(frame, verbose=False)
        result_frame = results[0].plot()

        # Ubah BGR ke RGB
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        stframe.image(result_frame, channels="RGB", use_container_width=True)

    cap.release()

# Mode Deteksi Gambar
if menu == "Deteksi Gambar":
    uploaded_image = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

        if st.button("🔍 Deteksi Objek"):
            with st.spinner("Sedang mendeteksi objek..."):
                result_image, labels = detect_image(image)
                st.image(result_image, caption="📌 Hasil Deteksi Objek", use_container_width=True)

                if labels:
                    st.success("🔎 Objek Terdeteksi:")
                    for i, label in enumerate(labels, 1):
                        st.markdown(f"- **Objek {i}:** {label}")
                else:
                    st.warning("❌ Tidak ada objek terdeteksi.")

# Mode Deteksi Video
elif menu == "Deteksi Video":
    uploaded_video = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        if st.button("▶️ Mulai Deteksi Video"):
            st.info("🎬 Menampilkan hasil deteksi objek secara real-time...")
            detect_video(tfile.name)
