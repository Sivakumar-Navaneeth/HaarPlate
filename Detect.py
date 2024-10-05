import cv2
import numpy as np
from skimage import filters
from skimage import io, color
import easyocr

reader = easyocr.Reader(['en'])

PlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

img = cv2.imread('Cars0.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = PlateCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors = 10, minSize=(25,25))

for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    plate = gray[y: y+h, x:x+w]
    gray[y: y+h, x:x+w] = plate

# cv2.imshow('plates',plate)

# edges = cv2.Canny(plate, 100, 200)
# # cv2.imshow('edges', edges)

# def sobel_edge_detection(image):
#     sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel along X-axis
#     sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel along Y-axis
#     sobel = cv2.magnitude(sobel_x, sobel_y)  # Magnitude of the gradient
#     return sobel

# def prewitt_edge_detection(image):
#     prewitt_x = filters.prewitt_h(image)  # Prewitt along X-axis
#     prewitt_y = filters.prewitt_v(image)  # Prewitt along Y-axis
#     prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)  # Magnitude of the gradient
#     return prewitt

# sobel_edges = sobel_edge_detection(plate)
# # cv2.imshow('Sobel', sobel_edges)

# prewitt_edges = prewitt_edge_detection(plate)
# # cv2.imshow('Prewitt', prewitt_edges)

def roberts_edge_detection(image):
    roberts = filters.roberts(image)  # Roberts edge detection
    return roberts

robert_edges = roberts_edge_detection(plate)
# cv2.imshow( 'Robert', robert_edges)

# Convert the plate to a format suitable for OCR
plate_for_ocr = cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)

# do OCR on the plate
text = reader.readtext(plate_for_ocr)

print("Detected Number:", text)

# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()