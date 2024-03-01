import cv2
import numpy as np

# Load the custom image template
template = cv2.imread("barcode.jpg")

# Convert the template to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply template matching
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Set a threshold for detection
    threshold = 0.4

    # Find locations where the template matches well
    loc = np.where(res >= threshold)

    # Check if a match is found
    if len(loc[0]) > 0:
        # Extract coordinates of the match
        x, y = loc[0][0], loc[1][0]

        # Draw a rectangle around the match
        cv2.rectangle(frame, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)

        # Capture the image
        cv2.imwrite("capture.jpg", frame)

        # Optional: Break the loop after capturing the image
        break

    # Display the frame
    cv2.imshow('frame', frame)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

