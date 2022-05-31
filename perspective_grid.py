import cv2

# Load image
def load_image(pathToImage):
    img = cv2.imread(pathToImage)
    return img

img = cv2.imread(
    r'image.jpg')

def draw_grid(img, color=(0, 255, 0)):
    # Add intersecting lines to find the focal point
    cv2.line(img, (0, 0),
             (int(img.shape[1]), int(img.shape[0])),
             color=color)

    cv2.imshow('grid', img)
    cv2.waitKey(0)
    return img

