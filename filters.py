import cv2
import numpy as np

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove Ruído
def remove_noise(image):
    return cv2.medianBlur(image,5)
    
# Segmentação
def thresholding(image, thresh):#return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

#Dilatação
def dilate(image):
    kernel = np.ones((5, 5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
        
#Erosão
def erode(image):
    kernel = np.ones((5, 5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#Abertura - dilate and erode
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#Alinhar
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 