import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

st.set_page_config(page_title="Prescription Reader", layout="centered")

st.title("💊 Prescription Reader (Smart)")
st.write("Upload a prescription image")

uploaded_file = st.file_uploader("Upload Prescription", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", width=400)

    # ---------------- PREPROCESS ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    st.image(thresh, caption="Processed Image", width=400)

    # ---------------- OCR ----------------
    config = r'--oem 3 --psm 6'
    raw_text = pytesseract.image_to_string(thresh, config=config)

    # ---------------- CLEAN TEXT ----------------
    text = raw_text.upper()

    # Fix common OCR mistakes
    replacements = {
        "0S": "OS",
        "5YP": "SYP",
        "SYF": "SYP",
        "MEFTPL": "MEFTAL",
        "LEVOLINS": "LEVOLIN",
        "CBLON": "DELCON",
        "QEH": "Q6H",
        "TO5": "TDS",
        "SMU": "3ML",
        "RO": "ML",
        "ORT": "RR"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    lines = text.split("\n")

    name = ""
    rr = ""
    medicines = []

    for line in lines:
        line = line.strip()

        if len(line) < 3:
            continue

        # NAME
        if "NAME" in line:
            name = line.split(":")[-1].strip()

        # RR extraction (clean)
        if "RR" in line:
            match = re.search(r'RR.*?(\d+)', line)
            if match:
                rr = match.group(1) + "/min"

        # MEDICINES (smart detection)
        if any(med in line for med in ["CALPOL", "DELCON", "LEVOLIN", "MEFTAL"]):
            medicines.append(line)

    # Remove duplicates
    medicines = list(set(medicines))

    # ---------------- DISPLAY ----------------
    st.subheader("👤 Patient Details")
    st.write(f"Name: {name if name else 'Not detected'}")
    st.write(f"RR: {rr if rr else 'Not detected'}")

    st.subheader("💊 Prescribed Medicines")

    if medicines:
        for med in medicines:
            st.write(f"• {med}")
    else:
        st.write("No medicines detected")

    # ---------------- RAW TEXT ----------------
    with st.expander("🔍 Raw OCR Text"):
        st.text(raw_text)
