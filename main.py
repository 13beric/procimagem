import cv2
import re
from processor import Processor

TESSERACT_PATH = r"C:\Users\carad\AppData\Local\Programs\Tesseract-OCR\Tesseract.exe"

#Abertura da imagem
original_img = cv2.imread('audi.jpg')

height, width, channels = original_img.shape



#TODO IDENTIFICAR TAMANHO DA IMAGEM E REDIMENSIONAR CONFORME NECESSIDADE
original_img = cv2.resize(original_img, None, fx=1600/width, fy=1000/height, interpolation=cv2.INTER_CUBIC)

processor = Processor()

#TODO: IDENTIFICAR E APLICAR FILTROS DE MANEIRA GERAL QUE NAO PREJUDIQUE IMAGENS ESPECIFICAS
preprocessed_img = processor.preprocess(original_img)

fragments_list = processor.text_localize(preprocessed_img, original_img)

# cv2.imshow('image', img)
# cv2.imshow('image preprocessed', preprocessed_img)


pattern = '[A-Z]{3}\d{4}'

for fragment in fragments_list:
        if  re.match(pattern,processor.text_recognize(fragment)):
                print(processor.text_recognize(fragment))

# cv2.waitKey(100000)
# cv2.destroyAllWindows()