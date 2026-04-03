import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

st.set_page_config(page_title="Prescription Reader", layout="centered")

st.title("💊 AI Prescription Reader")
st.markdown("Upload a prescription to extract structured medical information.")

uploaded_file = st.file_uploader("📤 Upload Prescription", type=["jpg", "png", "jpeg"])

# Known medicines
known_meds = ["CALPOL", "DELCON", "LEVOLIN", "MEFTAL"]

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="🖼 Uploaded Prescription", use_container_width=True)

    # ---------------- PREPROCESS ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    st.image(thresh, caption="🧪 Processed Image", use_container_width=True)

    # ---------------- OCR ----------------
    config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(thresh, config=config)

    st.subheader("🔍 Raw OCR Text")
    st.text(text)

    # ---------------- CLEAN ----------------
    text = text.upper()

    # Fix OCR mistakes
    text = text.replace("5YP", "SYP")
    text = text.replace("SYF", "SYP")
    text = text.replace("MEFTPL", "MEFTAL")
    text = text.replace("LEVOLINS", "LEVOLIN")
    text = text.replace("CBLON", "DELCON")
    text = text.replace("ORT", "RR")

    lines = text.split("\n")

    name = "Not Found"
    rr = "Not Found"
    medicines = []

    for line in lines:
        line = line.strip()

        if len(line) < 3:
            continue

        # NAME
        if "NAME" in line:
            match = re.search(r'NAME[:\s]+([A-Z]+)', line)
            if match:
                name = match.group(1)

        # RR
        if "RR" in line:
            match = re.search(r'(\d+)', line)
            if match:
                rr = match.group(1) + "/min"

        # MEDICINES
        for med in known_meds:
            if med in line:
                medicines.append(med)

    # Remove duplicates
    medicines = list(set(medicines))

    # ---------------- DISPLAY ----------------
    st.markdown("### 👤 Patient Details")
    st.write(f"**Name:** {name}")
    st.write(f"**RR:** {rr}")

    st.markdown("### 💊 Prescribed Medicines")

    if medicines:
        for med in medicines:
            st.write(f"• {med}")
    else:
        st.write("No medicines detected")
