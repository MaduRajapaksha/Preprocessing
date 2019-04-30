import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('1.jpg')
RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#grayscale
gray = cv2.cvtColor(RGB_img,cv2.COLOR_BGR2GRAY)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
   # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(0,0,255),5)
    cv2.waitKey(0)

plt.subplot(231),plt.imshow(RGB_img),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(gray, cmap="gray"),plt.title('gray')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(thresh),plt.title('second')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img_dilation),plt.title('DILATED')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(image),plt.title('MARKED')
plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(121),plt.imshow(image),plt.title('MARKED')
plt.xticks([]), plt.yticks([])
plt.show()