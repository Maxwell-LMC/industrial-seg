import numpy as np
import cv2
import matplotlib.pyplot as plt
import util

RESCALE_FACTOR = 2

# Load the image and resize
img = cv2.imread('Test_image/camera1_2023_08_01_17_20_36_936767.jpg')
dim = (img.shape[1] // RESCALE_FACTOR, img.shape[0] // RESCALE_FACTOR)
img = cv2.resize(img, dim)

side = img.shape[1] // 100

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
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_seg = img*mask2[:,:,np.newaxis]
    print(img_seg.shape)
    final_seg_img += img_seg

plt.imshow(final_seg_img)
plt.show()