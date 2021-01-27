import cv2
import numpy as np
import math

def im2double(img):
   """ Return a image in float64 format in a range of [0, 1] """
   info = np.iinfo(img.dtype) # Get the data type of the input image
   return img.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def im2uint8(img):
   """ This function converts the input image I1 to uint8. Input values in range [0,1] are scaled to [0,255],
    values grater than 1 are saturated to 255 """
   ttype = np.iinfo(np.uint8) # get target data type -> in this case uint8
   img = img * ttype.max
   img[img>ttype.max] = ttype.max
   return img.astype(ttype)

def shadding(img, imgbackground):
   """ This function compensates background by division """
   imgshadded = cv2.divide(im2double(img), im2double(imgbackground))
   imgshadded = im2uint8(imgshadded)

   return imgshadded

def imclearborder(imgBW, radius):
   """ Suppresses structures in image I that are lighter than their surroundings and that are connected
       to the image border. Use this function to clear the image border.  """
   imgBWcopy = imgBW.copy()
   contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   # Get dimensions of image
   imgRows = imgBW.shape[0]
   imgCols = imgBW.shape[1]

   contourList = [] # ID list of contours that touch the border

   # For each contour...
   for idx in np.arange(len(contours)):
       # Get the i'th contour
       cnt = contours[idx]

       # Look at each point in the contour
       for pt in cnt:
           rowCnt = pt[0][1]
           colCnt = pt[0][0]

           # If this is within the radius of the border
           # this contour goes bye bye!
           check1 = (rowCnt >= 0 and rowCnt <= radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
           check2 = (colCnt >= 0 and colCnt <= radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

           if check1 or check2:
               contourList.append(idx)
               break

   for idx in contourList:
       cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

   return imgBWcopy

#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
   """ Removes all connected components (objects) that have fewer than P pixels from the binary image BW, p
       roducing another binary image, BW2. This operation is known as an area opening. """
   imgBWcopy = imgBW.copy()
   image, contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   # For each contour, determine its total occupying area
   for idx in np.arange(len(contours)):
       area = cv2.contourArea(contours[idx])
       if (area >= 0 and area <= areaPixels):
           cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

   return imgBWcopy

def regionprops(imgbw):
   """ find contours in image """
   # Given a black and white image, first find all of its contours
   contours, hierarchy = cv2.findContours(imgbw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

   # calculate area of each contour and extract stats of the largest one
   area_vec = [cv2.contourArea(contours[i]) for i in range(len(contours))]
   idx = area_vec.index(max(area_vec))

   # get stats for contour with largest area
   cnt = contours[idx]
   # calculate centroid
   M = cv2.moments(cnt)
   cx = int(M['m10'] / M['m00'])
   cy = int(M['m01'] / M['m00'])
   # calculate orientation by determining smallest bounding rectangle
   rect = cv2.minAreaRect(cnt)
   # calculate orientation by calculating the fitting ellipse
   ellipse = cv2.fitEllipse(cnt)

   # print image with fitting ellipse
   if 0:
       cv2.imshow('dbg', cv2.ellipse(image, ellipse,(128,128,0),2))

   return (len(contours)-1), area_vec[idx], [cx,cy], rect, ellipse[2]

def rotate_around_point(vec, angle, point):
    """ Rotate an image around"""
    tx = point[0]
    ty = point[1]

    Trot = np.array([[math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0],
                     [-math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                     [0,0,1]])
    # may use np.matrix instead of array -> then * operator should work (arrays are no matricies
    Tshift1 = np.array([[1,0,0],
                        [0,1,0],
                        [-tx, -ty, 1]])

    Tshift2 = np.array([[1,0,0],
                        [0,1,0],
                        [tx, ty, 1]])

    T = Tshift1 @ Trot @ Tshift2
    vecrot = [vec[0], vec[1], 1] @ T

    return vecrot