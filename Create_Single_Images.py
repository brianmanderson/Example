__author__ = 'Brian M Anderson'
# Created on 2/17/2020

from Base_Deeplearning_Code.Make_Single_Images.Make_Single_Images_Class import run_main
from Return_Morfeus_Base_Paths import return_paths, os


def main():
    path, morfeus_path = return_paths()
    if os.path.exists(path):
        print(path)
    desired_output_spacing = (0.89648, 0.89648, 3)
    desired_output_spacing = (None, None, 1)
    extension = 32
    write_images = True
    re_write_pickle = False
    run_main(path, desired_output_spacing, extension, write_images, re_write_pickle, file_ext='_1mm')


if __name__ == '__main__':
    main()
