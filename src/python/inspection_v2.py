import cv2
import logging, os
import tensorflow as tf
import imutils as utils
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from utils import label_map_util

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
	
# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg')

#  machine learning stuff
labelmap_path = "../../config/labelmap.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()
model = tf.saved_model.load('../../model/saved_model')

# read all images from folders given in a list
for imagePath in glob.glob('../../img/All/*.jpg'):

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

        
