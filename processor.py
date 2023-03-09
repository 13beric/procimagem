import cv2
import numpy as np
import pytesseract

from filters import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\Tesseract.exe"

MIN_CONTOUR_AREA = 30

class Processor:
    
    def preprocess(self, image):
        #Transforma para escala de cinza
        gray = get_grayscale(image)
        #cv2.imshow('gray', gray)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        #cv2.imshow('blur', blur)

        cannyd = canny(blur)
        #cv2.imshow('cannyd', cannyd)

        return cannyd
    
    def text_localize(self, image, original):
        
        #Borra letras para gerar contornos maiores (frases / parágrafos)
        # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        # threshed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, rect_kernel)
        
        # dilated = dilate(threshed)
        
        #opened = opening(image)

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(original, contours, -1, (255, 255, 255), 1)

        # Converte para escala de cinza
        img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_cinza', img)
        
        #Binariza imagem
        _, img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
        cv2.imshow('img_binary', img)

        img = cv2.medianBlur(img, 5)
        cv2.imshow('img_desfoque', img)
        
        img = opening(img)
        cv2.imshow('img_erodida', img)


        text_blocks_images = []

        i = 0
        for c in contours:
            perimetro = cv2.arcLength(c, True)
            if perimetro > 300:
                aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
                if len(aprox) == 4:
                    (x, y, alt, lar) = cv2.boundingRect(c)
                    cv2.rectangle(original, (x, y), (x + alt, y + lar), (255, 0, 0), 2)
                    roi = img[y:(y + lar), x:x + alt]
                    text_blocks_images.append(roi)
                    cv2.imwrite(f'roi{i}.png', roi)
                    i += 1

        cv2.imshow('original', original)
                        
        return text_blocks_images
    
    def text_recognize(self, image):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

        return pytesseract.image_to_string(image, config=config)
        
        
