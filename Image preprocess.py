import pandas as pd
import numpy as np
from datetime import datetime

#for loading pdf filenames
import os
from os import listdir
from os.path import isfile, join

#for text region detection
import cv2
from pdf2image import convert_from_path
from PIL import Image

pdfPath = r'C:\project\pdf'
pdf_files = [f for f in listdir(f'{pdfPath}') if isfile(join(f'{pdfPath}', f))]
pdf_files = pdf_files[:10]
poppler = r'C:\project\poppler-21.03.0\Library\bin'
imagePath = r'C:\project\images'
colorPath = f'{imagePath}' + r'\color'
grayPath = f'{imagePath}' + r'\grayscale'


# Save pdf pages to images in "colorPath" directory
for i, f in enumerate(pdf_files):
    print(f'Now running file: {f}')
    images = convert_from_path(f'{pdfPath}\\{f}', poppler_path = poppler)
    for i, image in enumerate(images):
        print(f'  page {i+1}')
        img = np.asarray(image, dtype = np.uint8)
        img = Image.fromarray(cv2.resize(img, (1656, 2336)))
        filename = f.replace(".pdf", "").replace(".PDF", "")
        img.save(f'{colorPath}\{filename}_{i+1}.png')
        
# Transform images to grayscale and save to "grayPath" directory
baseIMG = listdir(f'{colorPath}')
for f in baseIMG:
    print(f'Now running file: {f}')
    img = cv2.imread(f'{colorPath}\{f}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{grayPath}\{f}', gray)

# Define function to modify image contrast and/or apply erosion
def preprocess(img, erode = 0, contrast = 0):
    img = img.astype(np.uint8)
    if (contrast > 0):
        alpha = 1 + contrast / 100
        img = cv2.convertScaleAbs(img, alpha = alpha, beta = 0)
    if erode:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations = 1)
        img = cv2.erode(img, kernel, iterations = 1)
    return img

# Apply combition of erosion and constrast and save to appropriate directory
for f in baseIMG:
    print(f'Now running file: {f}')
    img = cv2.imread(f'{grayPath}\{f}', 0)
    for erode in [0, 1]:
        for contrast in [20, 60]:
            folder = "ersn" + f'{erode}' + "cntrst" + f'{contrast}'
            target = f'{imagePath}\{folder}\{f}'
            out = preprocess(img, erode = erode, contrast = contrast)
            cv2.imwrite(target, out)
            
            
        












