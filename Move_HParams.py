__author__ = 'Brian M Anderson'
# Created on 5/12/2020

import os
import shutil

out_path = r'H:\Liver_Disease_Ablation\tensorboard'
input_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Disease_Ablation_Segmentation_Work\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training\Tensorboard'
for trial_id in os.listdir(input_path):
    if not os.path.exists(os.path.join(out_path,trial_id)):
        os.makedirs(os.path.join(out_path,trial_id))
    print(trial_id)
    path = os.path.join(input_path, trial_id)
    files = os.listdir(path)
    files = [i for i in files if i.find('events') == 0]
    for file in files:
        shutil.copy2(os.path.join(path,file),os.path.join(out_path,trial_id,file))
    if not os.path.exists(os.path.join(out_path,trial_id,'validation')):
        if 'validation' in os.listdir(path):
            shutil.copytree(os.path.join(path,'validation'),os.path.join(out_path,trial_id,'validation'))