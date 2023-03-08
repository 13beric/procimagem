import cv2
import numpy as np
import pytesseract

TESSERACT_PATH = r"C:\Users\ericb\AppData\Local\Programs\Tesseract-OCR\Tesseract.exe"
MIN_CONTOUR_AREA = 30

#Abertura da imagem
original_img = cv2.imread('imagem-teste2.jpg')
#Aumentando tamanho da imagem
img = cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

#TODO: Falta realizar o preprocessamento de imagens caso tenham ruído, estejam deterioradas, ou borradas
def identify_text_blocks(img):
        #Aplica filtro cinza na imagem
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Aplica blur na imagem
        blured = cv2.blur(gray, (5,5), 0)   
        # cv2.imshow('blured', blured)
 
        #Aplica filtro Gaussiano e P/B
        threshed = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # cv2.imshow('b/w', threshed)

        #Borra letras para gerar contornos maiores (frases / parágrafos)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        threshed_2 = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, rect_kernel)
        # cv2.imshow('b/w 2', threshed_2)

        #identifica contornos externos
        contours, _ = cv2.findContours(threshed_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

def extract_text_fragments(contours : list):
        
        #Parâmetros do grabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask = np.zeros(img.shape[:2], np.uint8)

        text_blocks_images = []

        for contour in contours:
                if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                        
                        #Pega medidas do contorno
                        x, y, w, h = cv2.boundingRect(contour)       
                        rect = (x, y, w, h)
                        
                        #Gera recorte do contorno
                        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                        gc = img.copy() * mask2[:, :, np.newaxis]

                        #Gera imagem do recorte e adiciona na lista
                        roi = gc[y:y + h, x:x + w]
                        text_blocks_images.append(roi)
                        
        return text_blocks_images
        
#TODO: Isso aqui precisa ser mais elaborado, identificando cada letra        
def identify_characters(fragment):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        return pytesseract.image_to_string(fragment)

contours = identify_text_blocks(img)
fragments_list = extract_text_fragments(contours)

for fragment in fragments_list:
        print(identify_characters(fragment))
        
cv2.waitKey(100000)
cv2.destroyAllWindows()