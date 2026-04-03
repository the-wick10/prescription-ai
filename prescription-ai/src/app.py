import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Prescription Reader", layout="centered")

st.title("💊 Prescription Reader (Improved)")
st.write("Upload a prescription image to extract details")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Upload Prescription", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.subheader("📷 Uploaded Image")
    st.image(image, width=400)

    # -------------------------------
    # Preprocessing (VERY IMPORTANT)
    # -------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    # Threshold
    _, thresh = cv2.threshold(sharp, 150, 255, cv2.THRESH_BINARY)

    st.subheader("🧠 Processed Image")
    st.image(thresh, width=400)

    # -------------------------------
    # OCR
    # -------------------------------
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # -------------------------------
    # CLEAN TEXT
    # -------------------------------
    lines = text.split("\n")

    name = ""
    rr = ""
    medicines = []

    for line in lines:
        line = line.strip()

        if len(line) < 3:
            continue

        # NAME
        if "NAME" in line.upper():
            name = line.split(":")[-1].strip()

        # RR
        if "RR" in line.upper():
            rr = line.strip()

        # MEDICINES
        if any(word in line.upper() for word in ["SYP", "TAB", "CAP", "SYRUP", "INJ"]):
            medicines.append(line)

    # -------------------------------
    # DISPLAY OUTPUT
    # -------------------------------
    st.subheader("👤 Patient Details")

    if name:
        st.write(f"Name: {name}")
    else:
        st.write("Name: Not detected")

    if rr:
        st.write(f"RR: {rr}")
    else:
        st.write("RR: Not detected")

    st.subheader("💊 Prescribed Medicines")

    if medicines:
        for med in medicines:
            st.write(f"• {med}")
    else:
        st.write("No medicines detected")

    # -------------------------------
    # RAW TEXT (for debugging)
    # -------------------------------
    with st.expander("🔍 Raw OCR Text"):
        st.text(text)
