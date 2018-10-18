import cv2
import numpy as np

from matplotlib import pyplot as plt

green = (0, 255, 0)

def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')
    plt.show()

def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    #easy function
    ellipse = cv2.fitEllipse(contour)
    #add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2,cv2.LINE_AA)
    return image_with_ellipse

def find_biggest_contour(image):
    image=image.copy()
    _,contours, hierarchy=cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes=[(cv2.contourArea(contour),contour) for contour in contours]
    biggest_contour=max(contour_sizes,key=lambda x:x[0])[1]

    mask=np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask,[biggest_contour],-1,255,-1)
    return  biggest_contour, mask

# def find_all(image):
#     image = image.copy()
#     _, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     mask = np.zeros(image.shape, np.uint8)
#     for i in contours:
#         cv2.drawContours(mask, i, -1, 255, -1)
#
#
#     show(mask)



def find_banana(image):

    #Chuyển sang RGB
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #resize
    max_dimension=max(image.shape)
    scale=700/max_dimension
    image=cv2.resize(image,None,fx=scale, fy=scale)

    #Gaussian filter
    image_blur=cv2.GaussianBlur(image,(5,5),0)

    #Chuyển sang HSV
    image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)

    # minimum yellow, max yellow
    min_yellow = np.array([20, 100, 100])
    max_yellow = np.array([30, 255, 255])

    mask=cv2.inRange(image_blur_hsv, min_yellow, max_yellow)

    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    mask_closed=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_cleaned=cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_banana_contour, mask_banana=find_biggest_contour(mask_cleaned)

    circled=circle_contour(image, big_banana_contour)

    return circled

img=cv2.imread('u.png')

result=find_banana(img)

show(result)

