import cv2

image = cv2.imread("in_img.jpg")
bilateral = cv2.bilateralFilter(image, 15, 75, 75)
cv2.imwrite("filtered_image_OpenCV.png", bilateral)
