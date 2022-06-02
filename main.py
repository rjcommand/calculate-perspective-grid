# This is a script to calculate a Canadian (perspective) grid for seafloor images.
# Following Wakefield & Genin 1987 (https://doi.org/10.1016/0198-0149(87)90148-8)

# Press ⌃R to execute

# Import module containing functions to create the grid
import perspective_grid
import numpy as np

import argparse
# Build the argument parser
parser = argparse.ArgumentParser()
# Argument for the path to the image
parser.add_argument("image_path", type=str,
                    help="path to the image for which a perspective grid is desired")
# Path to output image
parser.add_argument("output_image", type=str,
                    help="path to the location where the image with grid overlay should be saved")
# Set threshold level?
parser.add_argument('-th', '--threshold', type=int, default=100,
                    help='threshold level')
# Include guides?
parser.add_argument("-gu", "--guides", action='store_true', default=False,
                    help="show guides on image",)
# Include perspective grid?
parser.add_argument("-gr", "--grid", action='store_true', default=True,
                    help="show perspective grid on image")
# Make output verbose?
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                    help="increase output verbosity")
# Parse the arguments
args = parser.parse_args()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = perspective_grid.load_image(args.image_path)  # Formerly 'test_image.jpg'
    cropped = perspective_grid.remove_borders(img)
    cropped_with_curve = perspective_grid.draw_light_curve(cropped, th=args.threshold)
    img_with_grid = perspective_grid.overlay_grid(img=cropped, grid_interval=10, show_guides=args.guides, show_grid=args.grid)
    perspective_grid.save_image(args.output_image, img_with_grid)
    print('FOV area =', np.round(perspective_grid.calc_area()[9], 3), 'm^2')
