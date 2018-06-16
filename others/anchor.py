import cv2
import numpy as np 

#plot diagram of anchor box for my graduation project paper.
img = np.full((724, 724, 3), 255, dtype=np.uint8)
#same as Faster R-CNN provided.
loc = np.array([[316, 271, 407, 452],
       [271, 181, 452, 543],
       [181,   0, 543, 724],
       [298, 298, 426, 426],
       [234, 234, 490, 490],
       [106, 106, 618, 618],
       [271, 316, 452, 407],
       [181, 271, 543, 452],
       [  0, 181, 724, 543]])

for i in range(loc.shape[0]):
    print(loc[i][0], loc[i][1])
    cv2.rectangle(img, (loc[i][0], loc[i][1]), (loc[i][2], loc[i][3]), (0, 0, 0), 5)

cv2.imshow('img', img)
cv2.waitKey(0)

img = cv2.resize(img, (400, 400))
cv2.imwrite("anchor.jpg", img)