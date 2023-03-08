import cv2
import numpy as np
import pytesseract

from filters import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

TESSERACT_PATH = r"C:\Users\ericb\AppData\Local\Programs\Tesseract-OCR\Tesseract.exe"
MIN_CONTOUR_AREA = 30

class Processor:
    
    def preprocess(self, image):
        
        #Transforma para escala de cinza
        gray = get_grayscale(image)
        
        # Remoção de Ruídos
        # noise = remove_noise(gray)
        
        #Preto e Branco
        threshold = thresholding(gray)

        return threshold
    
    def text_localize(self, image):
        
        #Borra letras para gerar contornos maiores (frases / parágrafos)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        threshed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, rect_kernel)
        
        dilated = dilate(threshed)
        
        opened = opening(dilated)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        text_blocks_images = []
        for contour in contours:
                if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                        
                        #Pega medidas do contorno
                        x, y, w, h = cv2.boundingRect(contour)       
                        roi = image[y:y + h, x:x + w]
                        text_blocks_images.append(roi)
                        
        return text_blocks_images
    
    def text_recognize(self, image):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        return pytesseract.image_to_string(image)
        
        
