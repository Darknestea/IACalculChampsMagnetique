import os
from os import listdir

from Utils.constants import RAW_PATH, EXPERIMENT_TEST

# This file can be used to batch rename images in a folder to 0.png ... n.png
if __name__ == '__main__':
    my_path = f"{RAW_PATH(EXPERIMENT_TEST)}RealBeamSlices\\"

    files = listdir(my_path)
    for id_image, file_name in enumerate(files):
        old_path = my_path + "\\" + file_name
        new_path = my_path + "\\" + str(id_image) + ".png"
        os.rename(old_path, new_path)
        # print(old_path)
        # print(new_path, "\n")
