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

    # ---------------- PREPROCESS ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    st.image(thresh, caption="🧪 Processed Image", use_container_width=True)

    # ---------------- ROI OCR ----------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text = ""

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # filter noise
        if w > 50 and h > 20:
            roi = thresh[y:y+h, x:x+w]

            roi_text = pytesseract.image_to_string(
                roi,
                config='--oem 3 --psm 7'
            )

            text += roi_text + "\n"

    lines = text.split("\n")

    st.subheader("📋 Extracted Information")

    name = "Not Found"
    rr = "Not Found"
    medicines = []

    for line in lines:
        line = line.strip()

        if len(line) > 3:
            line = clean_line(line)

            # NAME extraction
            if "NAME" in line:
                match = re.search(r'NAME[:\s]+([A-Z]+)', line)
                if match:
                    name = match.group(1)

            # RR extraction
            if "RR" in line:
                match = re.search(r'(\d+)', line)
                if match:
                    rr = match.group(1) + "/min"

            # MEDICINE detection
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

    # remove duplicates
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

    # ---------------- RAW TEXT ----------------
    with st.expander("🔍 Raw OCR Text"):
        st.text(text)
