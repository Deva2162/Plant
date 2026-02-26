import cv2

# Load the image
img = cv2.imread('/mnt/data/Screenshot 2025-07-08 163946.png')

# Resize for better visualization
img = cv2.resize(img, (500, 700))

# Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert the gray image
inv_gray = 255 - gray

# Apply Gaussian blur
blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)

# Blend using dodge technique
def dodge(front, back):
    result = cv2.divide(front, 255 - back, scale=256)
    return result

sketch = dodge(gray, blur)

# Show the sketch
cv2.imshow("Lord Hanuman Sketch", sketch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the sketch
cv2.imwrite('hanumanji_sketch_output.png', sketch)
