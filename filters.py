import cv2
import numpy as np

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove Ruído
def remove_noise(image):
    return cv2.medianBlur(image,5)
    
# Segmentação
def thresholding(image, thresh, mode):
    return cv2.threshold(image, thresh, 255, mode)[1]

#Dilatação
def dilate(image):
    kernel = np.ones((10, 10),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
        
#Erosão
def erode(image): 
    kernel = np.ones((10, 10),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)