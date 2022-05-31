# This is a script to calculate a Canadian (perspective) grid for seafloor images.
# Following Wakefield & Genin 1987 (https://doi.org/10.1016/0198-0149(87)90148-8)

# Press ⌃R to execute

# Import module containing functions to create the grid
import perspective_grid

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = perspective_grid.load_image('test_image.jpg')
    cropped = perspective_grid.remove_borders(img)
    perspective_grid.draw_grid(img=cropped)

