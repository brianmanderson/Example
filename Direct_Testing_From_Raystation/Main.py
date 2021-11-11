__author__ = 'Brian M Anderson'
# Created on 5/26/2020

# First, export the file as MetaData from Raystation

#Export_MetaData_From_Raystation.py

#Next, convert it into .nii.gz
convert_mhd_to_nifti = False
if convert_mhd_to_nifti:
    from Deep_Learning.Direct_Testing_From_Raystation.Raystation_to_Niftii import mhd_to_nifii
    mhd_to_nifii(path = r'H:\Liver_Disease_Ablation\test_export')
print('Test')
#Now make it into a tf record
make_tf_record = False
if make_tf_record:
    path = r'H:\Liver_Disease_Ablation\test_export'
    from Pre_Processing.Make_Single_Images.Make_TFRecord_Class import write_tf_record, os
    from Pre_Processing.Make_Single_Images.Image_Processors_Module.Image_Processors_TFRecord import *
    image_processors = [Normalize_to_annotation(annotation_value_list=[1,2], mirror_max=True),
                        Box_Images(wanted_vals_for_bbox=[1,2],power_val_z=2**3, power_val_r=2**3, power_val_c=2**3),
                        Distribute_into_3D(mirror_small_bits=True, chop_ends=False, desired_val=2)]
    write_tf_record(path, record_name='Test', image_processors=image_processors,
                    is_3D=True, rewrite=True, shuffle=True, thread_count=1)

desc='Raystation'
model_path = r'H:\Liver_Disease_Ablation\Keras\TF2_3D_Fully_Atrous_Variable_Cube_Training\Models\Trial_ID_199\model'
create_prediction = False
if create_prediction:
    from Deep_Learning.Evaluate_Model.Write_Predictions import create_prediction_files
    create_prediction_files(is_test=True, desc=desc, model_path=model_path, cache=False)
