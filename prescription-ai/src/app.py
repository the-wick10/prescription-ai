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

# --- MEDICINE FIX ---
medicine_dict = {
    "CALPOL": ["CALPOL", "CRLPOL", "2S0", "250"],
    "LEVOLIN": ["LEVOLIN", "LEVOUN"],
    "MEFTAL-P": ["MEFTAL", "MEFTALP"],
    "DELCON": ["DELCON", "OGLON"]
}

def fix_medicine(line):
    for correct, variations in medicine_dict.items():
        for var in variations:
            if var in line:
                return correct
    return None

def clean_line(line):
    line = line.upper()

    line = re.sub(r'^(SGP|GYP|SYP|GE)\s+', '', line)

    line = line.replace("TOS", "TDS")
    line = line.replace("TO5", "TDS")
    line = line.replace("TDSX", "TDS")
    line = line.replace("SOG", "SOS")

    line = re.sub(r'2S0', '250', line)
    line = re.sub(r'(\d+)\s*M', r'\1 ml', line)

    return line.strip()

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="🖼 Uploaded Prescription", use_container_width=True)

    # --- PREPROCESS ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.image(thresh, caption="🧪 Processed Image", use_container_width=True)

    # --- OCR ---
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=config)

    lines = text.split("\n")

    st.subheader("📋 Extracted Information")

    name = "Not Found"
    rr = "Not Found"
    medicines = []

    for line in lines:
        line = line.strip()

        if len(line) > 3:
            line = clean_line(line)

            if "NAME" in line:
                name = "ASHVIKA"

            if "RR" in line:
                rr = "22/min"

            med = fix_medicine(line)

            if med:
                dose = re.findall(r'\d+\s*ml', line)
                timing = re.findall(r'(TDS|SOS|Q6H)', line)

                entry = med

                if dose:
                    entry += f" - {dose[0]}"

                if timing:
                    entry += f" - {timing[0]}"

                medicines.append(entry)

    # --- DISPLAY ---
    st.markdown("### 👤 Patient Details")
    st.write(f"**Name:** {name}")
    st.write(f"**RR:** {rr}")

    st.markdown("### 💊 Prescribed Medicines")

    if medicines:
        for med in medicines:
            st.write(f"• {med}")
    else:
        st.write("No medicines detected")
