import numpy as np
import cv2
import matplotlib.pyplot as plt
import util
import sys


#click on the positions for segmenting, press "Enter" after finishing selecting.


RESCALE_FACTOR = 2

# Load the image and resize, also sharpen
img = cv2.imread('Test_image/camera1_2023_08_01_17_20_36_936767.jpg')
dim = (img.shape[1] // RESCALE_FACTOR, img.shape[0] // RESCALE_FACTOR)
img = cv2.resize(img, dim)

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
img = cv2.filter2D(img, -1, kernel)

side = img.shape[1] // 70

fig = plt.figure()

rects = []

def onclick(event):
    ix, iy = event.xdata, event.ydata
    rects.append(util.generate_rect_from_point(round(ix), round(iy), side=side))

def key_press(event):
    if event.key == 'enter':
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_disconnect(cid2)
        plt.close()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid2 = fig.canvas.mpl_connect('key_press_event', key_press)

plt.imshow(img),plt.show()

final_seg_img = np.zeros(img.shape,np.uint8)

for rect in rects:
    print(rect)
    mask = util.circle_mask_generation(rect=rect, diameter=side//1.8, img_size=img.shape)

    #showing the mask
    plt.imshow(mask)
    plt.show()
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_seg = img*mask2[:,:,np.newaxis]
    final_seg_img += img_seg

plt.imshow(final_seg_img)
plt.show()