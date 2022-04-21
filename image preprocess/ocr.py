from os import listdir, mkdir
from os.path import join, isdir, exists
import time
import cv2
import pandas as pd
import pytesseract
from pytesseract import Output

tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract

# Get directories in "images" directory and names of png files in one of those
basePath = r'C:\project'
imgPath = join(basePath, "images")
dirNam = [d for d in listdir(imgPath) if isdir(join(imgPath, d))]
imgNam = [f for f in listdir(join(imgPath, dirNam[0])) if f.split(".")[1] == "png"]

# Create a "text" directory with subdirectories as in "images" directory
txtPath = join(basePath, "text")
if not exists(txtPath): mkdir(txtPath)
for d in dirNam: 
    if not exists(join(txtPath, d)): mkdir(join(txtPath, d))

# Define function to produce text from image given the image's path
def OCR(imagePath):
    img = cv2.imread(imagePath)
    start = time.time()
    d = pytesseract.image_to_data(img, output_type = Output.DICT)
    ocrTime = time.time() - start
    return pd.DataFrame.from_dict(d), ocrTime
    
# For all png files in the subdirectories of "images" folder, save text
# in the corresponding subdirectories of "text" folder
for d in dirNam:
    print(f'\nOCR in "{d}" directory took:')
    for f in imgNam:
        inputPath  = join(imgPath, d, f)
        outputPath = join(txtPath, d, f.replace(".png", ".csv"))
        text, ocrTime = OCR(inputPath)
        print(f'\t {ocrTime:.2f} sec for: {f.replace(".png", "")}')
        text.to_csv(outputPath, index = False)
        
