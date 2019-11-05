
import cv2
import numpy as np


def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))




img = cv2.imread('../test/test.jpg')

# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = histogram_equalize(img)

# convert the YUV image back to RGB format
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)

cv2.waitKey(0)
