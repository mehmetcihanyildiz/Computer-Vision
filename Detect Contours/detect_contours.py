import cv2
import numpy as np


 #   Applies preprocessing steps to the image by:
 #   1. Converting the image to grayscale.
 #   2. Applying GaussianBlur to reduce noise.

def preprocess_image(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return blurred

# Detects and draws contours in the image.
def find_and_draw_contours(image_path):

    # Preprocess the image
    blurred = preprocess_image(image_path)
    if blurred is None:
        return

    # Apply Canny edge detection
    edged = cv2.Canny(blurred, 70, 105)

    # Dilate and erode for better contour detection
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contour filtration based on area
    min_contour_area = 1
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw contours on the original image using ApproxPolyDP
    image = cv2.imread(image_path)
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    # Display results
    cv2.imshow('Original Image', cv2.imread(image_path))
    cv2.imshow('Canny Edges', edged)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Number of Contours found = " + str(len(contours)))

# Example usage
find_and_draw_contours('happy-pup-1.png')
