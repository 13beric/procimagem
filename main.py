import cv2
from processor import Processor

#Abertura da imagem
original_img = cv2.imread('audi.jpg')

height, width, channels = original_img.shape

original_img = cv2.resize(original_img, None, fx=1600/width, fy=1600/width, interpolation=cv2.INTER_CUBIC)

processor = Processor()

processed_images = processor.image_preprocess(original_img)

fragments_list = processor.plate_identify(original_img, processed_images)

processor.plate_recongnize(fragments_list)

cv2.waitKey(10000)
cv2.destroyAllWindows()