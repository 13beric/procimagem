import cv2
import pytesseract
import re

from filters import get_grayscale, remove_noise, thresholding, dilate, erode, opening, canny, deskew, match_template

from env import *

MIN_CONTOUR_AREA = 300

class Processor:
    
    #Possibilidades de threshold (imagem, placa) para melhor se adaptar a imagem
    thresholds = [[50, 30], [85, 50], [140, 100]]
    
    def image_preprocess(self, image):
        
        #Escala de Cinza
        gray = get_grayscale(image)
        
        #Desfoque
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        #Gera lista de dicts com a imagem com cada threshold e o respectivo threshold que tem que ser aplicado à placa
        image_dicts_list = []
        for threshold in self.thresholds:
            threshimage = thresholding(blur, threshold[0])
            image_dicts_list.append({
                    'image': threshimage,
                    'plate_threshold': threshold[1]
                })
            cv2.imshow(f'Image{threshold[1]}', threshimage)

        imageteste = cv2.adaptiveThreshold(remove_noise(blur), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        image_dicts_list.append({
            'image': imageteste,
            'plate_threshold': 50
        })
        cv2.imshow(f'Imageteste', imageteste)


        return image_dicts_list
    
    def plate_identify(self, original_img, processed_images : list):
        
        possible_plates = []
        
        for image_dict in processed_images:
            contours, _ = cv2.findContours(image_dict['image'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)      
            
            for c in contours:
                perimetro = cv2.arcLength(c, True)
                if perimetro > MIN_CONTOUR_AREA:
                    aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
                    if len(aprox) == 4:
                        (x, y, alt, lar) = cv2.boundingRect(c)
                        roi = original_img[y:(y + lar), x:x + alt]
                        possible_plates.append({
                            'plate_image': roi,
                            'threshold': image_dict['plate_threshold']
                        })
                        
        return possible_plates
        
    def plate_recongnize(self, fragments : list):
        
        results = []
        i = 0
        for fragment_dict in fragments:
            
            resized_plate = cv2.resize(fragment_dict['plate_image'], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)    
            
            img = get_grayscale(resized_plate)
            
            img = remove_noise(img)

            
            img = thresholding(img, fragment_dict['threshold'])
      
            img = erode(img)

            #img = dilate(img)



            cv2.imwrite(f'roi{i}.jpg', img)
            i += 1

            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        
            recognized_text = pytesseract.image_to_string(img, config=config)
            
            pattern = '[A-Z]{3}\d{4}'
            a = re.search(pattern, recognized_text)
            
            if recognized_text and a is not None:
                results.append(recognized_text)
                
        for result in set(results):
            print(result) 
