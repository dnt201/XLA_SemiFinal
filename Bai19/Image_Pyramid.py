
#------------------------------------#
# Author: Yueh-Lin Tsou              #
# Update: 7/16/2019                  #
# E-mail: hank630280888@gmail.com    #
#------------------------------------#

"""-----------------------------------
- Image pyramid
    - Gaussian pyramid
    - Laplacian pyramid
-----------------------------------"""

# Import OpenCV Library, numpy and command line interface
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

# ------------------- Function to do Gaussian image pyramid ------------------- #
def Gaussian_Image_pyramid(img):

    lower_1 = cv2.pyrDown(img)
    lower_2 = cv2.pyrDown(lower_1)
    lower_3 = cv2.pyrDown(lower_2)
    lower_4 = cv2.pyrDown(lower_3)

    higher_4 = cv2.pyrUp(lower_4)
    higher_3 = cv2.pyrUp(higher_4)
    higher_2 = cv2.pyrUp(higher_3)
    higher_1 = cv2.pyrUp(higher_2)

    # show result
    plt.subplot(231),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(cv2.cvtColor(lower_2, cv2.COLOR_BGR2RGB))
    plt.title('Down Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(cv2.cvtColor(lower_4, cv2.COLOR_BGR2RGB))
    plt.title('Down Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(cv2.cvtColor(lower_4, cv2.COLOR_BGR2RGB))
    plt.title('Process Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(cv2.cvtColor(higher_3, cv2.COLOR_BGR2RGB))
    plt.title('Up Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(cv2.cvtColor(higher_1, cv2.COLOR_BGR2RGB))
    plt.title('Up Image'), plt.xticks([]), plt.yticks([])

    plt.show()


# -------------------------- main -------------------------- #
if __name__ == '__main__':
    # read one input from terminal
    # (1) command line >> python Image_Pyramid.py -i dog.jpg

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(ap.parse_args())

    # Read image
    image = cv2.imread(args["image"])

    ## Functions
    Gaussian_Image_pyramid(image)

