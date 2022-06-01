import cv2
import numpy as np


def point_trans_contrast_dynamic(image, m, e):
    image = image.astype(np.float32) / 255
    trans_image = 1/(1+(m/image)**e)
    norm_image = cv2.normalize(trans_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    trans_image = norm_image.astype(np.uint8)
    return trans_image

def point_trans_gamma(image, const, gamma):
    image = image.astype(np.float32) / 255
    trans_image = (image**gamma)*const
    norm_image = cv2.normalize(trans_image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    trans_image = norm_image.astype(np.uint8)
    return trans_image

def crop(image ,x_a, x_b, y_a, y_b):
    image_croped = image[x_a:x_b, y_a:y_b] 
    return image_croped

def convert(frame): #transformations used for lerning model
    frame = cv2.resize(frame, (180, 180))                
    return frame
    
def prepare(frame):
    return frame.reshape((-1, 97200))