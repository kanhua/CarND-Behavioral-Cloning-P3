import numpy as np
import cv2

IMAGE_HEIGHT=66
IMAGE_WIDTH=200


def clahe_image(image):
    yuv_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
    yuv_img[:,:,2]= clahe.apply(yuv_img[:,:,2])
    n_img = cv2.cvtColor(yuv_img,cv2.COLOR_HSV2RGB)
    return n_img


def resize(image):
    """
    Resize the image to IMAGE_WIDTHxIMAGE_HEIGHT

    :param image:
    :return:
    """
    n_image=cv2.resize(image,dsize=(IMAGE_WIDTH,IMAGE_HEIGHT))
    return n_image

def crop(image):

    cropped_img = image[60:-25, :]

    return cropped_img


def to_yuv(image):

    yuv_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    return yuv_img


def preprossing(image):

    n_img=crop(image)
    n_img=resize(n_img)
    n_img=to_yuv(n_img)

    return n_img



