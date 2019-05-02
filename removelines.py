
import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('image1.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image ,5)

#
ret,th1 = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

plt.subplot(141),plt.imshow(image,  cmap="gray"),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(th1,  cmap="gray"),plt.title('binary')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(th2,  cmap="gray"),plt.title('mean')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(th3,  cmap="gray"),plt.title('adaptive gausian')
plt.xticks([]), plt.yticks([])
plt.show()