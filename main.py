import cv2

from processor import Processor

TESSERACT_PATH = r"C:\Users\ericb\AppData\Local\Programs\Tesseract-OCR\Tesseract.exe"

#Abertura da imagem
original_img = cv2.imread('imagem-teste.jpg')

#TODO IDENTIFICAR TAMANHO DA IMAGEM E REDIMENSIONAR CONFORME NECESSIDADE
img = cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

processor = Processor()

#TODO: IDENTIFICAR E APLICAR FILTROS DE MANEIRA GERAL QUE NAO PREJUDIQUE IMAGENS ESPECIFICAS
preprocessed_img = processor.preprocess(img)

fragments_list = processor.text_localize(preprocessed_img)

# cv2.imshow('image', img)
# cv2.imshow('image preprocessed', preprocessed_img)

for fragment in fragments_list:
        print(processor.text_recognize(fragment))

cv2.waitKey(10000)
cv2.destroyAllWindows()