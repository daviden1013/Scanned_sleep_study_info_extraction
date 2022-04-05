# Scanned_sleep_study_info_extraction
This is an information extraction OCR-NLP project. We use a random sampled set of 955 sleep study reports (as images in PDF) from  University of Texas Medical Branch to develop a data pipeline for extracting numeric values: apnea hypopnea index (AHI) and oxygen saturation (SaO2). 
**For HIPPA reasons, there is NO data in this repositary**. The purpose is to document the source code for future reference.

## Requirements ##
- Tesseract==4.0.0
- OpenCV==4.5.2
- nltk==3.6.2
- Scikit-learn==0.24.2
- TensorFlow==2.2.0
- Gensim==4.0.1
- transformers==4.6.1
- PyTorch==1.9.0

## Method Overview ##
![Method flowchart](https://github.com/daviden1013/Scanned_sleep_study_info_extraction/blob/main/flowchart.png)

## Image preprocessing ##
We extract images pages from the PDF files followed by image preprocessing using the Open Source Computer Vision Library (OpenCV, version 4.5.2). We first convert the 3-channel color images to 1-channel gray-scale to reduce computation commplexity, then dilate and erode each character by 1 iteration of transformation. The dilation process shrinks objects (characters) and results in the removal of small noise dots, while the erosion process converts the image back to the original scale. Finally, we increased the contrast by 20% thus background noises caused by scanning were further removed. 
Code: [image preprocess](https://github.com/daviden1013/Scanned_sleep_study_info_extraction/tree/main/image%20preprocess)

## Optical Character Recognition (OCR) ##
We apply Tesseract OCR (version 4.0.0) via pytesseract to locate and extract machine-readable text from the preprocessed images. The output for each image is a mapping of extracted words and positions in pixels. We performed a data quality visual inspection by programmatically drawing outlines of each word onto the original images using the positions with OpenCV. 

## Text processing ##
Candidate words for AHI and SaO2 values are identified using a regular expression for words that match “[0-9.,%]+”. For each numeric value,  a segment of 10 words on each side of the candidate (21 words total) is used for context. 
Code: [text processing](https://github.com/daviden1013/Scanned_sleep_study_info_extraction/tree/main/text%20processing)

## Text classification ##
At this point, the information extraction problem can be cast into a three-way classification task: whether the candidate numeric value is an AHI value, a SaO2 value, or neither. Each instance has a set of position indicators obtained from OCR, the page number from which the numeric value was extracted, a floating-point representation of the numeric value, and a segment of 21 words. Our human review did not include information on positions where the AHI and SaO2 values were found. We assigned labels by matching the recorded AHI and SaO2 numbers to each of the numeric values in the document. Therefore, as a limitation, we cannot rule out false positives if some other numeric values in the same report happened to be the same number as the AHI or SaO2, though we suspect this to be quite rare. 
In our main experiment, we construct and train two types of NLP models: bag-of-word models and deep learning-based sequence models. 

![Deep learning models](https://github.com/daviden1013/Scanned_sleep_study_info_extraction/blob/main/model.png)

Code: [training](https://github.com/daviden1013/Scanned_sleep_study_info_extraction/tree/main/training)

## Model evaluation ##
To examine the NLP models, we split the reports into a 70% (N=669) development set and a 30% test set (N=286). For the bag-of-word models, we performed 5-fold cross-validation using the development set to search for an optimal parameter set that maximizes the validation accuracy. We then re-trained each model with the entire development set given the optimal parameters. For the deep learning-based sequence models, due to the high computation, we further split the 70% development set with a 6:1 ratio into a training set (N=574) and a validation set (N=95). We saved checkpoints after each epoch and used the validation set to select the best checkpoint as our final model, based on cross-entropy loss. The BiLSTM model was trained using a batch size of 64, with Adam optimization with a learning rate of 2e-4 for 100 epochs. BERT and ClinicalBERT were fine-tuned using a batch size of 64, with Adam optimization with a learning rate of 2e-6 for 100 epochs. 
After training, the final models were evaluated with the test set. We evaluated at the segment level using recall, precision, and the area under the receiver operating characteristic curve (AUROC) for AHI and SaO2. To better assess our final goal for information extraction, we also evaluated at the document level. The numeric value in a document with the highest probability for AHI (or SaO2) was selected to represent the document. 
Code: [evaluation](https://github.com/daviden1013/Scanned_sleep_study_info_extraction/tree/main/evaluation)

