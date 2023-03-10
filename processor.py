import cv2
import pytesseract
import re

from filters import get_grayscale, remove_noise, thresholding, dilate, erode

from env import TESSERACT_PATH

MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 1500

class Processor:
    thresholds = [90, 140]
    
    def image_preprocess(self, image):
        
        gray = get_grayscale(image)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        images_list = []
        for threshold in self.thresholds:
            threshimage = thresholding(blur, threshold, cv2.THRESH_BINARY)
            images_list.append(threshimage)
        return images_list
    
    def plate_identify(self, original_img, preprocessed_images : list):
        possible_plates = []
        
        for image in preprocessed_images:
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        
            for c in contours:
                perimetro = cv2.arcLength(c, True)
                if perimetro > MIN_CONTOUR_AREA and perimetro < MAX_CONTOUR_AREA:
                    aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
                    if len(aprox) == 4:
                        (x, y, alt, lar) = cv2.boundingRect(c)
                        roi = original_img[y+5:(y + lar)-5, x+10:x + alt - 10]
                        possible_plates.append(roi)
        
        return possible_plates
        
    def plate_recongnize(self, fragments : list, id):
        results = []
        
        for fragment in fragments:
            
            resized_plate = cv2.resize(fragment, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)    
            
            img = get_grayscale(resized_plate)
            
            img = remove_noise(img)

            img = thresholding(img, 0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            img = dilate(img)
            img = erode(img)

            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        
            recognized_text = pytesseract.image_to_string(img, config=config)
            
            patterns = []
            patterns.append('[A-Z]{3}\d{1}[A-Z]{1}\d{2}')
            patterns.append('[A-Z]{3}\d{4}')

            for pattern in patterns:
                a = re.search(pattern, recognized_text)
                if a is not None:
                    results.append(a.group())
                    cv2.imwrite(f'output/placa_processada_{id}.jpg', img)
                    cv2.imwrite(f'output/placa_{id}.jpg', fragment)


                
        return set(results)
