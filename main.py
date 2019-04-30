import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

img = cv2.imread('1.jpg', cv2.IMREAD_UNCHANGED)
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#h,s,v = cv2.split(hsv)

BLUE = [255,0,0]


#cv2.imread(img);
#filter
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

#constant
constant = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

#remove specific colour range
upper = np.array([255,248,248])  #-- Upper range --
lower = np.array([102,35,0])  #-- Lowerrange --
mask = cv2.inRange(img, lower, upper)
res = cv2.bitwise_and(img, img, mask= mask)  #-- Contains pixels having the gray color--

#remove colours except white

data = np.array(res)
converted = np.where(data == 255,255,255)
wht = Image.fromarray(converted.astype(np.uint8))

#keep the text only
mask = cv2.inRange(img, (0,0,0), (220, 220, 220))
dst1 = cv2.bitwise_and(img, img, mask=mask)

plt.subplot(151),plt.imshow(img),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(constant),plt.title('constant')
plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(res),plt.title('remove color range')
plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow(wht),plt.title('keep specific color')
plt.xticks([]), plt.yticks([])
plt.subplot(155),plt.imshow(dst1),plt.title('keep text only')
plt.xticks([]), plt.yticks([])
plt.show()

