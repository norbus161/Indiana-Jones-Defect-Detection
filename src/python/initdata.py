import cv2
import numpy as np

def initdata():
    # read template image for template matching and leave it unchanged to extract alpha channel
    # alpha channel is used to create binary mask
    img = cv2.imread('../../img/templates/template.png', cv2.IMREAD_UNCHANGED)
    imgg = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    alpha_channel = (np.around(img[:,:,3]/255)).astype(np.uint8)
    indy, indx = np.where(alpha_channel)
    templatecenter = (np.round(np.mean(indx)), np.round(np.mean(indy)))

    # remove hat from the mask, hat is too big and is causing errors
    img_mask = cv2.imread('../../img/templates/mask_hat.png',0)
    img_mask[img_mask < 1] = 0
    img_mask[img_mask >= 1] = 1
    templatemask = cv2.bitwise_and(alpha_channel, (1-img_mask))

    # store template dependent data into a dict
    template = {'img': img,
                'imgg': imgg,
                'center':templatecenter,
                'mask': templatemask}

    # load defects and store them into a dict, which is firstly allocated
    # for each defect, a mask, path to images, and a name is stored
    defects = {}
    lmask = cv2.imread('../../img/templates/mask_lhand.png',0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    rmask = cv2.imread('../../img/templates/mask_rhand.png', 0)
    rmask[rmask < 1] = 0
    rmask[rmask >= 1] = 1
    defects['hand'] = {'mask': cv2.bitwise_or(lmask,rmask), 'dir':'5-NoHand/', 'name': 'Hand missing'}
    lmask = cv2.imread('../../img/templates/mask_larm.png', 0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    rmask = cv2.imread('../../img/templates/mask_rarm.png', 0)
    rmask[rmask < 1] = 0
    rmask[rmask >= 1] = 1
    defects['arm'] = {'mask': cv2.bitwise_or(lmask,rmask), 'dir':'7-NoArm/', 'name': 'Arm missing'}
    lmask = cv2.imread('../../img/templates/mask_lleg.png', 0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    rmask = cv2.imread('../../img/templates/mask_rleg.png', 0)
    rmask[rmask < 1] = 0
    rmask[rmask >= 1] = 1
    defects['leg'] = {'mask': cv2.bitwise_or(lmask,rmask), 'dir':'3-NoLeg/', 'name': 'Leg missing'}
    lmask = cv2.imread('../../img/templates/mask_hat.png', 0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    defects['hat'] = {'mask':lmask, 'dir':'1-NoHat/', 'name': 'Hat missing'}
    lmask = cv2.imread('../../img/templates/mask_face.png', 0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    defects['face print'] = {'mask':lmask, 'dir':'2-NoFace/','name': 'Face print missing'}
    lmask = cv2.imread('../../img/templates/mask_body.png', 0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    defects['body print'] = {'mask':lmask, 'dir':'4-NoBodyPrint/', 'name': 'Body print missing'}

    lmask = cv2.imread('../../img/templates/mask_head.png', 0)
    lmask[lmask < 1] = 0
    lmask[lmask >= 1] = 1
    defects['head'] = {'mask': lmask, 'dir': '6-NoHead/', 'name': 'Head missing'}

    return template, defects