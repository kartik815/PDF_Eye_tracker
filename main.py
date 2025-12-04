import fitz
import cv2
import numpy as np

def highlight_word(img, words, index, color=(0, 0, 255), thickness=3):
    """Draw highlight on a specific word index."""
    frame = img.copy()
    if 0 <= index < len(words):
        x0, y0, x1, y1, text, _, _, _ = words[index]
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
    return frame


# Open PDF and get first page
doc = fitz.open(r"C:\Users\Kartik\OneDrive\Desktop\study\CSE\DSA\PDF_eye_tracking_project\test.pdf")
page = doc[0]

# Render page to an image
pix = page.get_pixmap()
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if pix.n == 4:  # RGBA
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
else:  # RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Extract words
words = page.get_text("words")

index = 0
paused = False
delay = 500  # milliseconds between words

while True:
    frame = highlight_word(img, words, index)
    cv2.imshow("PDF Words Highlight", frame)

    key = cv2.waitKey(delay) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('d'):  # next word
        if index < len(words) - 1:
            index += 1
        paused = True  # stop auto-scroll when manually moving
    elif key == ord('a'):  # previous word
        if index > 0:
            index -= 1
        paused = True
    elif key == ord('p'):  # pause/resume
        paused = not paused
        print("Paused" if paused else "Resumed")

    if not paused:
        index = (index + 1) % len(words)


cv2.waitKey(0)
cv2.destroyAllWindows()
