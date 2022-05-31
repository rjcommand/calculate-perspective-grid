# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import perspective_grid

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = perspective_grid.load_image('image.jpg')
    perspective_grid.draw_grid(img=img)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
