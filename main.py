# This is a script to calculate a Canadian (perspective) grid for seafloor images.
# Following Wakefield & Genin 1987 (https://doi.org/10.1016/0198-0149(87)90148-8)

# Press ⌃R to execute

# Import module containing functions to create the grid
import perspective_grid

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = perspective_grid.load_image('test_image.jpg')
    cropped = perspective_grid.remove_borders(img)
    img_with_grid = perspective_grid.overlay_grid(img=cropped, grid_interval=10, show_guides=True, show_grid=False)
    perspective_grid.save_image('test_image_with_guides.png', img_with_grid)
