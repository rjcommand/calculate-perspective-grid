import cv2
import numpy as np


# Load image
def load_image(pathToImage):
    img = cv2.imread(pathToImage)
    return img


# Remove black side bars from image
def remove_borders(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)

    crop = img[y:y+h, x:x+w]

    cv2.imwrite('test_image_cropped.png', crop)

    return crop


# Find point P (the focal point in the image)
# This assumes that the camera is calibrated so that diagonal lines drawn from corner to corner intersect at P
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

print line_intersection((A, B), (C, D))

# Draw the perspective grid
def draw_grid(img, color=(0, 255, 0)):
    # Add intersecting lines to find the focal point
    cv2.line(img, (0, 0),
             (int(img.shape[1]), int(img.shape[0])),
             color=color)

    cv2.line(img, (0, int(img.shape[0])),
             (int(img.shape[1]), 0),
             color=color)

    focal_point = line_intersection(line1=((0, 0), (int(img.shape[1]), int(img.shape[0]))),
                                    line2=((0, int(img.shape[0])), (int(img.shape[1]), 0)))
    cv2.circle(img, focal_point, radius=0, color=(0, 0, 255), thickness=1)

    cv2.imshow('grid', img)
    cv2.waitKey(0)
    return img
