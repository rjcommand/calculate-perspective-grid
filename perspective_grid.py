import math
import cv2
import numpy as np


# - - - - - - - - - - - - - - - - - #
# Image loading and pre-processing  #
# - - - - - - - - - - - - - - - - - #

# Load image
def load_image(path_to_image):
    img = cv2.imread(path_to_image)
    return img


# Remove black side bars from image
def remove_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = img[y:y + h, x:x + w]

    cv2.imwrite('test_image_cropped.png', crop)

    return crop


# - - - - - - - - - - - - - - - - - - - #
# Calculate & display perspective grid  #
# - - - - - - - - - - - - - - - - - - - #

# Find point P (the focal point in the image)
# This assumes that the camera is calibrated so that diagonal lines drawn from corner to corner intersect at P
# Following: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
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


# Draw the perspective grid
# Definitions of points and angles from Wakefield & Genin (1987)
# O     = nodal point of camera's lens
# N     = nadir point, where plumb line from nodal point intersects substratum
# ON    = camera's height above the seafloor
# p     = principal point, intersection of optical axis with image
# theta = inclination of optical axis (below the horizontal line): angle(POR)
# alpha = vertical acceptance angle / angle of view: angle(BOU)
# beta  = horizontal acceptance angle / angle of view: angle(FOE)
# base line = the lower edge of the image, ef
# horizontal lines = Canadian grid lines parallel to the base line (ef)
# meridian lines = Canadian grid lines, all converging on a single "vanishing point"
# principal line = Canadian grid line (bu) which intersects the principal point (p) and is perpendicular to ef

# Calculate horizontal lines (vertical scale)
# Angles
theta = 22.2
alpha = 47.4
beta = 60.4
# Distances
ON = 44.11
BA = 20  # This is the grid interval we want
# Distance from the base line to the principal point
bp = (ON * math.tan(90 - (alpha / 2) - theta)) - (ON * math.tan(90 - theta))

# Distance to the first horizontal line in the image, representing a distance of 20cm from baseline on the seafloor
ba = bp * (1 + (math.tan(theta - math.atan(ON / (BA + ON * (1 / math.tan(theta + alpha / 2)))))) / math.tan(alpha / 2))

var = (10, 20, 30, 40, 50)


def calc_h_line(img, theta=22.2, alpha=47.4, camera_height=44.11, grid_interval=100):
    # Convert degree angle to radians
    theta = theta * (math.pi / 180)  # Angle of incidence with the seafloor (input as degrees)
    alpha = alpha * (math.pi / 180)  # Vertical acceptance angle (input as degrees)
    ON = camera_height  # This is the height of the camera (cm)
    BA = grid_interval  # This is the grid interval we want
    # Distance from the base line to the principal point
    bp = ((ON * math.tan((90 * math.pi/180) - theta)) - (ON * math.tan((90 * math.pi/180) - (alpha / 2) - theta)))

    # Distance to the first horizontal line in the image, representing a distance of 20cm from baseline on the seafloor
    ba = bp * (1 + (math.tan(theta - math.atan(ON / (BA + ON * (math.cos(theta + alpha / 2) / math.sin(theta + alpha / 2)))))) /
               math.tan(alpha / 2))

    # Return the points with respect to the image frame
    # return ba
    return [(0, int(img.shape[0] - ba)), (int(img.shape[1]), int(img.shape[0] - ba))]


# Calculate vertical lines (horizontal scale)
def calc_v_line(ON, theta, alpha, beta):
    OB = ON / (math.sin(theta + (alpha / 2)))

    EF = 2 * OB * math.tan(beta / 2)

    Shb = (EF * math.sin(theta + alpha / 2)) / (2 * ON * math.tan(beta / 2))

    return EF, Shb


# Draw the perspective grid ontop of the image
def draw_grid(img, color=(0, 255, 0)):
    # Add intersecting lines to find the focal point
    # Line sf
    cv2.line(img,
             pt1=(0, 0),
             pt2=(int(img.shape[1]), int(img.shape[0])),
             color=color)
    # Line et
    cv2.line(img,
             pt1=(0, int(img.shape[0])),
             pt2=(int(img.shape[1]), 0),
             color=color)

    # Find the principal point (where the two diagonals intersect)
    principal_point = line_intersection(line1=((0, 0), (int(img.shape[1]), int(img.shape[0]))),
                                        line2=((0, int(img.shape[0])), (int(img.shape[1]), 0)))
    # Draw the focal point
    cv2.circle(img, center=(int(principal_point[0]), int(principal_point[1])),
               radius=4, color=(0, 0, 255), thickness=5)

    # Add horizontal and vertical guides through the focal point
    # Line gh
    # Point 'g' and 'h' are the left and right edges of the image, respectively
    cv2.line(img,
             pt1=(0, int(principal_point[1])),
             pt2=(int(img.shape[1]), int(principal_point[1])),
             color=color)
    # Line bu
    # Point 'b' and 'u' are the bottom and top edges of the image, respectively
    cv2.line(img,
             pt1=(int(principal_point[0]), 0),
             pt2=(int(principal_point[0]), int(img.shape[0])),
             color=color, thickness=2)
    # Distances between reciprocal edges and p are equal (gp = ph; bp = pu)
    # Horizontal lines
    hlines = np.array([calc_h_line(img, theta=22.2, alpha=47.4, camera_height=44.11, grid_interval=x)
                       for x in (600, 800)])
    cv2.polylines(img, hlines, isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.imshow('grid', img)
    cv2.waitKey(0)
    return img
