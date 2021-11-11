__author__ = 'Brian M Anderson'
# Created on 2/17/2020

import os
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Return_Paths import find_base_dir, find_raid_dir


def return_paths():
    try:
        base = r'\\mymdafiles\di_data1'
        base_path = r'H:\Liver_Disease_Ablation'
        if not os.path.exists(base_path):
            base_path = os.path.join(base,
                                     r'Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Niftii_Data')
        os.listdir(base_path)
        morfeus_drive = os.path.abspath(
            os.path.join(base, 'Morfeus', 'BMAnderson', 'Modular_Projects', 'Liver_Disease_Ablation_Segmentation_Work'))
    except:
        base = find_raid_dir()
        base_path = os.path.join(base, 'Liver_GTV_Ablation')
        morfeus_drive = os.path.abspath(
            os.path.join(find_base_dir(), 'Morfeus', 'BMAnderson', 'Modular_Projects', 'Liver_Disease_Ablation_Segmentation_Work'))
    excel_path = os.path.join(morfeus_drive, 'Parameters_List.xlsx')
    return base_path, morfeus_drive, excel_path


if __name__ == '__main__':
    pass
