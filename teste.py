import cv2
import pytesseract
from env import *

original_img = cv2.imread('testando.png')

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

print(pytesseract.image_to_string(original_img, config=config))
