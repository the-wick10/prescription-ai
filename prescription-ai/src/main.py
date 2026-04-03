import cv2
import pytesseract
import re

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    keywords = ["SYP", "LEVOLIN", "MEFTAL", "CALPOL", "RR", "Age", "Name"]

    for line in lines:
        line = line.strip()

        # remove unwanted symbols
        line = re.sub(r'[^a-zA-Z0-9 .]', '', line)

        # fix OCR mistakes
        line = line.replace("S4yP", "SYP")
        line = line.replace("MEFTALP", "MEFTAL-P")

        if len(line) > 4:
            for key in keywords:
                if key.lower() in line.lower():
                    cleaned.append(line)
                    break

    return "\n".join(cleaned)


# ---------------- LOAD IMAGE ----------------
print("Loading image: sample2.jpg")

image = cv2.imread("/Users/hemanth/Desktop/prescription-ai/data/sample2.jpg")

if image is None:
    print("Error: Image not found")
    exit()

print("Image loaded successfully")

# ---------------- CROP HANDWRITTEN AREA ----------------
h, w, _ = image.shape
cropped = image[int(h*0.25):int(h*0.85), :]

# ---------------- PREPROCESS ----------------
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# ---------------- MERGE TEXT INTO LINES ----------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# ---------------- FIND CONTOURS ----------------
contours, _ = cv2.findContours(
    dilated,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# ---------------- SORT TOP TO BOTTOM ----------------
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

output = cropped.copy()
final_text = ""

# ---------------- PROCESS EACH REGION ----------------
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    if w > 60 and h > 20:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        roi = cropped[y:y+h, x:x+w]

        # resize for better OCR
        roi = cv2.resize(roi, None, fx=2, fy=2)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)

        # OCR config (single line)
        config = "--oem 3 --psm 7"
        text = pytesseract.image_to_string(thresh_roi, config=config)

        cleaned = clean_text(text)

        if cleaned != "":
            final_text += cleaned + "\n"

# ---------------- SHOW DETECTION ----------------
cv2.imshow("Handwritten Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------- FINAL OUTPUT ----------------
print("\n--- FINAL EXTRACTED TEXT ---\n")
print(final_text)