import cv2
from matplotlib import pyplot as plt

import os
import imutils as utils
import glob
import initdata as init
import random # only used to return random labels

# hook here your function to inspect image and return label for the detected defect
def inspect_image(img, defects):
    img_processed = img
    predicted_label = random.randrange(0,7)
    return img_processed, predicted_label


# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg')

template, defects = init.initdata()

do_plot = False # Enable plotting of images which are processed

y_true, y_pred = [], [] # container for ground truth label and predicted label


#for class_label, defect_type in enumerate(  ):
#imageDir = "../../img/" + defects[defect_type]['dir']
imageDir = "../../img/All/" 

# read all images from folders given in a list
for imagePath in glob.glob(imageDir + "*.jpg"):

    img = cv2.imread(imagePath)
    if img is None:
        print("Error loading: " + imagePath)
        # end this loop iteration and move on to next image
        continue

    """
    ... perform defect detection here
    
    """

    img = utils.shadding(img, imgbackground)
    pathname = "../../img-corrected-background/" + os.path.basename(imagePath)
    print(pathname)
    cv2.imwrite(pathname, img)

#from sklearn.metrics import accuracy_score, confusion_matrix

#print("Accuracy: ", accuracy_score(y_true, y_pred))
#print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

