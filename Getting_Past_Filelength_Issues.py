__author__ = 'Brian M Anderson'
# Created on 2/12/2020

import os, shutil


def down_folder(path, base_input, base_output):
    files = []
    folders = []
    for root, folders, files in os.walk(path):
        break
    event_files = [i for i in files if i.find('event') == 0]
    if event_files:
        after_base = path.split(base_input)[1]
        for file in event_files:
            new_path_base = base_output + after_base
            if not os.path.exists(new_path_base):
                os.makedirs(new_path_base)
            shutil.copy(os.path.join(path,file),os.path.join(new_path_base,file))
    for folder in folders:
        down_folder(os.path.join(path,folder),base_input, base_output)


path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work\Keras\3D_Atrous_new_livernorm\Tensorboard\4_layers\1_atrous_blocks\2_atrous_rate\1_max_atrous_blocks\32_filters\32_max_filters\mask_pred\Adam_opt_name\threshold_to_0'
out_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work\Keras\3D_Atrous_new_livernorm\Tensorboard\Default_Architecture\Adam_opt_name\threshold_to_0\linear_cycle_scale_mode'
down_folder(path, path, out_path)