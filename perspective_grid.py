import math
import cv2
import numpy as np
from scipy.interpolate import splprep, splev  # For contour smoothing function
import random as rng
rng.seed(12345)

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


def calc_h_line(img, theta=22.2, alpha=47.4, camera_height=44.11, grid_interval=20):
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

    # Find the principal point
    principal_point = line_intersection(line1=((0, 0), (int(img.shape[1]), int(img.shape[0]))),
                                        line2=((0, int(img.shape[0])), (int(img.shape[1]), 0)))

    # Calculate the distance from the horizontal line to the principal point as a ratio
    hline_dist = principal_point[1] * (ba / bp)

    # Return the points with respect to the image frame
    hline_points = [(0, int(img.shape[0] - hline_dist)), (int(img.shape[1]), int(img.shape[0]- hline_dist))]
    return hline_points


# Calculate vertical lines (horizontal scale)
def calc_v_line(img, ON=44.11, theta=22.2, alpha=47.4, beta=60.4, grid_side='left', grid_interval=10):
    theta = theta * (math.pi / 180)
    alpha = alpha * (math.pi / 180)
    beta = beta * (math.pi / 180)
    delta = (90 * math.pi/180) - (alpha / 2) - theta

    bp = ((ON * math.tan((90 * math.pi/180) - theta)) - (ON * math.tan((90 * math.pi/180) - (alpha / 2) - theta)))

    # Distance from camera to base line, orthogonal to base line
    OB = ON / (math.sin(theta + (alpha / 2)))

    # Width of bottom of field of view
    EF = 2 * OB * math.tan(beta / 2)
    ef = 2 * math.tan(beta / 2) * (ON / math.cos(delta))
    # Scaling factor for hb (horizontal scale at the base line
    Shb = (ef * math.sin(theta + alpha / 2)) / (2 * ON * math.tan(beta / 2))

    OP = ON / math.sin(theta)

    GH = 2 * OP * math.tan(beta / 2)
    # Scaling factor for hp (horizontal scale at the principal point)
    Shp = (ef * math.sin(theta)) / (2 * ON * math.tan(beta/2))

    # Find the principal point
    principal_point = line_intersection(line1=((0, 0), (int(img.shape[1]), int(img.shape[0]))),
                                        line2=((0, int(img.shape[0])), (int(img.shape[1]), 0)))
    # Distance from left edge of frame to center (principal point along the base line)
    eb = ef / 2
    # Distance from the center to the grid interval
    bk = grid_interval * Shb

    pj = grid_interval * Shp
    distance_to_vanishing = bk * (bp / (bk - pj))

    # Calculate the distance along the baseline point to line bu (line perpendicular to the center baseline) as a ratio
    vline_base_point_left = principal_point[0] - principal_point[0] * ((bk / eb) * Shb)
    vline_base_point_right = principal_point[0] + principal_point[0] * ((bk / eb) * Shb)

    # Calculate the distance from the grid point to the vertical line through the principal point as a ratio
    vline_p_point_left = principal_point[0] - principal_point[0] * ((pj / eb) * Shp)
    vline_p_point_right = principal_point[0] + principal_point[0] * ((pj / eb) * Shp)

    # Get the points with respect to the image frame
    vlines_left = (vline_base_point_left, img.shape[0]), (vline_p_point_left, distance_to_vanishing)
    vlines_right = (vline_base_point_right, img.shape[0]), (vline_p_point_right, distance_to_vanishing)

    # Return each left and right side
    if grid_side == 'left':
        return vlines_left
    elif grid_side == 'right':
        return vlines_right


def draw_guides(img, pt_color=(0, 0, 255), guide_color=(0, 255, 0)):
    # Add intersecting lines to find the focal point
    # Line sf
    cv2.line(img,
             pt1=(0, 0),
             pt2=(int(img.shape[1]), int(img.shape[0])),
             color=guide_color)
    # Line et
    cv2.line(img,
             pt1=(0, int(img.shape[0])),
             pt2=(int(img.shape[1]), 0),
             color=guide_color)

    # Find the principal point (where the two diagonals intersect)
    principal_point = line_intersection(line1=((0, 0), (int(img.shape[1]), int(img.shape[0]))),
                                        line2=((0, int(img.shape[0])), (int(img.shape[1]), 0)))
    # Draw the focal point
    cv2.circle(img, center=(int(principal_point[0]), int(principal_point[1])),
               radius=4, color=pt_color, thickness=5)

    # Add horizontal and vertical guides through the focal point
    # Line gh
    # Point 'g' and 'h' are the left and right edges of the image, respectively
    cv2.line(img,
             pt1=(0, int(principal_point[1])),
             pt2=(int(img.shape[1]), int(principal_point[1])),
             color=guide_color)
    # Line bu
    # Point 'b' and 'u' are the bottom and top edges of the image, respectively
    cv2.line(img,
             pt1=(int(principal_point[0]), 0),
             pt2=(int(principal_point[0]), int(img.shape[0])),
             color=guide_color, thickness=2)

    return img


def draw_grid(img, color=(255, 255, 255), grid_interval=10):
    # Find the principal point (where the two diagonals intersect)
    principal_point = line_intersection(line1=((0, 0), (int(img.shape[1]), int(img.shape[0]))),
                                        line2=((0, int(img.shape[0])), (int(img.shape[1]), 0)))
    # Line bu
    # Point 'b' and 'u' are the bottom and top edges of the image, respectively
    cv2.line(img,
             pt1=(int(principal_point[0]), 0),
             pt2=(int(principal_point[0]), int(img.shape[0])),
             color=color, thickness=1, lineType=cv2.LINE_AA)
    # Distances between reciprocal edges and p are equal (gp = ph; bp = pu)
    # Horizontal lines
    h_intervals = np.arange(grid_interval, 160, grid_interval, dtype=object).tolist()
    hlines = np.array([calc_h_line(img, theta=22.2, alpha=47.4, camera_height=44.11, grid_interval=x)
                       for x in h_intervals])
    # Draw the lines
    cv2.polylines(img, hlines, isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    # Vertical lines
    v_intervals = np.arange(grid_interval, 100, grid_interval, dtype=object).tolist()
    vlines_left = np.array([calc_v_line(img, ON=44.11, theta=22.2, alpha=47.4, beta=60.7, grid_side='left',
                                        grid_interval=x)
                            for x in v_intervals])

    vlines_right = np.array([calc_v_line(img, ON=44.11, theta=22.2, alpha=47.4, beta=60.7, grid_side='right',
                                         grid_interval=x)
                             for x in v_intervals])

    # Draw the vertical lines
    cv2.polylines(img, np.int32(vlines_left), isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.polylines(img, np.int32(vlines_right), isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    return img


# Draw the perspective grid ontop of the image
def overlay_grid(img, guide_color=(0, 255, 0), grid_color=(255, 255, 255), grid_interval=10, show_guides=True, show_grid=True):
    if show_guides & show_grid:
        draw_grid(img=img, color=grid_color, grid_interval=grid_interval)
        draw_guides(img=img, guide_color=guide_color)
    elif show_grid:
        draw_grid(img=img, color=grid_color, grid_interval=grid_interval)
    elif show_guides:
        draw_guides(img=img, guide_color=guide_color)

    # Show the image
    cv2.imshow('grid', img)
    cv2.waitKey(0)
    return img


# Function to smooth contours
# From: https://agniva.me/scipy/2016/10/25/contour-smoothing.html
def smooth_contours(cnts, original_img):
    smoothened = []
    for contour in cnts:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

    # Overlay the smoothed contours on the original image
    cv2.drawContours(original_img, smoothened, -1, (255, 255, 255), 2)

    return original_img


def draw_light_curve(img, th=100, adaptive=False):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Global thresholding
    _, thresh = cv2.threshold(grey, th, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    _, thresh_otsu = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ad_thresh = cv2.adaptiveThreshold(grey, th, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    # Show only the masked regions in the image
    masked = cv2.bitwise_and(img, img, mask=thresh)
    masked_ad = cv2.bitwise_and(img, img, mask=ad_thresh)

    kernel = np.ones((5, 5), np.uint8)
    thresh_erode = cv2.erode(thresh, None, iterations=6)
    thresh_dilate = cv2.dilate(thresh_erode, kernel, iterations=7)


    # Find contours
    contours, hierarchy = cv2.findContours(thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    img_cnts = smooth_contours(contours, original_img=img)

    for cnt in contours:
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])

    if rightmost[1] != leftmost[1]:
        rightmost_fixed = tuple((rightmost[0], leftmost[1]))
        rightmost = tuple(rightmost_fixed)

    cv2.circle(img, topmost, radius=2, thickness=5, color=(0, 0, 255))
    cv2.circle(img, leftmost, radius=2, thickness=5, color=(0, 0, 255))
    cv2.circle(img, rightmost, radius=2, thickness=5, color=(0, 0, 255))
    cv2.line(img, pt1=topmost, pt2=leftmost, color=(0, 255, 0))
    cv2.line(img, pt1=topmost, pt2=rightmost, color=(0, 255, 0))

    return thresh, print(leftmost, topmost, rightmost[0])

def save_image(filepath, img):
    cv2.imwrite(filepath, img)