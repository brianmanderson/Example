import os
import time, datetime


def down_folder(input_path):
    files = []
    folders = []
    for root, folders, files in os.walk(input_path):
        break
    event_files = [i for i in files if i.find('event') == 0]
    if event_files:
        if 'atrous_rate' not in input_path:
            atrous_block = input_path.split('_atrous_blocks')[0].split('\\')[-1] + '_atrous_blocks'
            out_path = input_path.replace(atrous_block,os.path.join(atrous_block,'2_atrous_rate'))
            # if not os.path.exists(out_path):
            #     os.makedirs(out_path)
            # for file in event_files:
            #     os.rename(os.path.join(input_path,file),os.path.join(out_path,file))
        elif '1_atrous_rate' in input_path:
            print(input_path)
            for file in event_files:
                created = datetime.datetime.strptime(time.ctime(os.path.getctime(os.path.join(input_path, file))), "%a %b %d %H:%M:%S %Y")
                if created.day < 31 or created.hour < 13:
                    out_path = input_path.replace('1_atrous_rate','2_atrous_rate')
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    os.rename(os.path.join(input_path, file), os.path.join(out_path, file))
        xxx = 1
    for folder in folders:
        down_folder(os.path.join(input_path,folder))
    files = []
    folders = []
    for root, folders, files in os.walk(input_path):
        break
    if not files and not folders:
        os.rmdir(input_path)
    return None


path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation\Keras\3D_Atrous_livernorm\Tensorboard'
down_folder(path)