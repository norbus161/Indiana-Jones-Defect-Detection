from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
from matplotlib import pyplot as plt

import tensorflow as tf
import imutils as utils
import glob
import initdata as init
import random  # only used to return random labels

from utils import label_map_util

# hook here your function to inspect image and return label for the detected defect


def inspect_image(img, defects):
    img_processed = img
    predicted_label = random.randrange(0, 7)
    return img_processed, predicted_label


# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg')

template, defects = init.initdata()

# Enable plotting of images which are processed
do_plot = False  

# container for ground truth label and predicted label
y_true, y_pred = [], []  

#  machine learning stuff
labelmap_path = "../../config/labelmap.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()
model = tf.saved_model.load('../../model/saved_model')


for class_label, defect_type in enumerate(defects):
    imageDir = "../../img/" + defects[defect_type]['dir']

    # read all images from folders given in a list
    for imagePath in glob.glob(imageDir + "*.jpg"):

        img = cv2.imread(imagePath)
        if img is None:
            print("Error loading: " + imagePath)
            # end this loop iteration and move on to next image
            continue

        """ ----------------------------------------------------- 
        ... perform defect detection here """

        # 1. compensate the background of the image
        img = utils.shadding(img, imgbackground)

        # 2. perform a call to the object detection network
        prediction_dict = utils.run_inference_for_single_image(model, img)
        print(prediction_dict)
        # 3. analzye the prediction dictionary 
        # if prediction_dict.

        """ ----------------------------------------------------- """

        img_processed, predicted_label = inspect_image(img, defects)
        y_pred.append(predicted_label)
        y_true.append(class_label)  # append real class label to true y's

        if (do_plot):
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.axis("off")
            ax1.set_title(imagePath)
            ax2.imshow(img_processed, cmap='gray')
            ax2.axis("off")
            ax2.set_title("Processed image")
            plt.show()


print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
