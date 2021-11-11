__author__ = 'Brian M Anderson'
# Created on 5/26/2020
import SimpleITK as sitk
import os
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


def mhd_to_nifii(path=r'H:\Liver_Disease_Ablation\test_export'):
    exam = sitk.ReadImage(os.path.join(path,'Examination.mhd'))
    liver = sitk.ReadImage(os.path.join(path,'Liver.mhd'))
    ablation = sitk.ReadImage(os.path.join(path,'Ablation.mhd'))
    liver = liver>255/2
    ablation = ablation>255/2
    combined = sitk.GetArrayFromImage(liver) + sitk.GetArrayFromImage(ablation)
    combined = sitk.GetImageFromArray(combined)
    combined.SetOrigin(exam.GetOrigin())
    combined.SetDirection(exam.GetDirection())
    combined.SetSpacing(exam.GetSpacing())
    return None


if __name__ == '__main__':
    pass
