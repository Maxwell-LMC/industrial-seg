import numpy as np
import matplotlib.pyplot as plt
import copy

def generate_rect_from_point(x,y,side,type="screw"):
    if type == "screw":
        return (x-side//2, y-side//2, side, side)
    
def circle_mask_generation(rect, diameter, img_size, type="screw"):
    side = rect[2]
    center = (rect[1] + side//2, rect[0] + side //2)
    temp = np.zeros(shape=img_size[:2])
    rect_temp = np.full(shape=(side, side), fill_value=3, dtype=np.uint8)
    temp[rect[1]:rect[1]+side,rect[0]:rect[0]+side] += rect_temp
    mask = gen_circle(temp, center, diameter)
    mask = mask.astype(np.uint8)
    return mask

def gen_circle(img: np.ndarray, center: tuple, diameter: int) -> np.ndarray:


  
    """
        Creates a matrix of ones filling a circle.
    """

    # gets the radious of the image
    radious  = diameter//2

    # gets the row and column center of the image
    row, col = center 

    # generates theta vector to variate the angle
    theta = np.arange(0, 360)*(np.pi/180)

    # generates the indexes of the column
    y = (radious*np.sin(theta)).astype("int32") 

    # generates the indexes of the rows
    x = (radious*np.cos(theta)).astype("int32") 

    # with:
    # img[x, y] = 1
    # you can draw the border of the circle 
    # instead of the inner part and the border. 

    # centers the circle at the input center
    rows = x + (row)
    cols  = y + (col)

    # gets the number of rows and columns to make 
    # to cut by half the execution
    nrows = rows.shape[0] 
    ncols = cols.shape[0]

    # makes a copy of the image
    img_copy = copy.deepcopy(img)

    # We use the symmetry in our favour
    # does reflection on the horizontal axes 
    # and in the vertical axes

    for row_down, row_up, col1, col2 in zip(rows[:nrows//4],
                            np.flip(rows[nrows//4:nrows//2]),
                            cols[:ncols//4],
                            cols[nrows//2:3*ncols//4]):
    
        img_copy[row_up:row_down, col2:col1] = 1

 
    return img_copy

# center = (30,40)
# ones = np.zeros((100, 100))
# diameter = 30

# circle = gen_circle(ones, center, diameter)
# plt.imshow(circle)
# plt.show()

# mask = circle_mask_generation(rect = (25,25,50,50), diameter = 20, img_size = (100,100))
# print(mask[50,50])
# plt.imshow(mask)
# plt.show()