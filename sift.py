import numpy as np
import cv2

image = cv2.imread("object.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
image = cv2.drawKeypoints(gray, kp, image)
cv2.imwrite('sift_keypoints.png', image)
