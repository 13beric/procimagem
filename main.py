import cv2
from processor import Processor

for i in range(1, 14):
    original_img = cv2.imread(f'input/carro{i}.jpg')

    height, width, channels = original_img.shape

    original_img = cv2.resize(original_img, None, fx=1600/width, fy=1600/width, interpolation=cv2.INTER_CUBIC)

    processor = Processor()

    processed_images = processor.image_preprocess(original_img)

    fragments_list = processor.plate_identify(original_img, processed_images)

    results = processor.plate_recongnize(fragments_list, f'carro{i}')
    print(f"Imagem: carro{i}.jpg | Placa reconhecida: {results if not len(results) == 0 else 'Placa não reconhecida'}")
    
# cv2.waitKey(100000)
# cv2.destroyAllWindows()