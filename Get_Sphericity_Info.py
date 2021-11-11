__author__ = 'Brian M Anderson'
# Created on 3/31/2020

import SimpleITK as sitk
from threading import Thread
from multiprocessing import cpu_count
from queue import *
from radiomics import cShape
import os, time
import numpy as np
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors import get_bounding_box_indexes
from Deep_Learning.Base_Deeplearning_Code.Send_Email_To_Phone.Send_Email_Module import Send_Email_Class
import pandas as pd


class Fill_Binary_Holes(object):
    def __init__(self, kernel_radius=(5,5,1)):
        self.BinaryfillFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryfillFilter.SetFullyConnected(True)
        self.BinaryfillFilter = sitk.BinaryMorphologicalClosingImageFilter()
        self.BinaryfillFilter.SetKernelRadius(kernel_radius)
        self.BinaryfillFilter.SetKernelType(sitk.sitkBall)

    def process(self, pred_handle):
        output = self.BinaryfillFilter.Execute(pred_handle)
        return output


class Threshold_and_Expand(object):
    def __init__(self, seed_threshold_value=0.8, lower_threshold_value=0.2):
        self.threshold_value = seed_threshold_value
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.Connected_Threshold = sitk.ConnectedThresholdImageFilter()
        self.Connected_Threshold.SetLower(lower_threshold_value)
        self.Connected_Threshold.SetUpper(2)
        self.stats = sitk.LabelShapeStatisticsImageFilter()

    def process(self, prediction):
        thresholded_image = sitk.BinaryThreshold(prediction, lowerThreshold=self.threshold_value)
        connected_image = self.Connected_Component_Filter.Execute(thresholded_image)
        self.stats.Execute(connected_image)
        seeds = [self.stats.GetCentroid(l) for l in self.stats.GetLabels()]
        seeds = [thresholded_image.TransformPhysicalPointToIndex(i) for i in seeds]
        self.Connected_Threshold.SetSeedList(seeds)
        output = self.Connected_Threshold.Execute(prediction)
        return output


class Create_Sphericity(object):
    def process(self,out_dict, path=r'H:\Liver_Disease_Ablation\Train\Overall_mask_LiTs_y8.nii.gz'):
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        RelabelComponent = sitk.RelabelComponentImageFilter()
        temp_dict = {'Group':[],'Patient_ID': [], 'Tumor_ID': [], 'Sphericity': [], 'Volume_cm3':[],'Voxels':[]}
        group = path.split('\\')[-2]
        annotation = sitk.ReadImage(path)
        pat_id = path.split('_y')[1].split('.nii')[0]
        print(pat_id)
        liver = sitk.BinaryThreshold(annotation, lowerThreshold=1,upperThreshold=3)
        label_image = Connected_Component_Filter.Execute(sitk.BinaryThreshold(annotation, lowerThreshold=2))
        base_mask = sitk.GetArrayFromImage(RelabelComponent.Execute(label_image))
        z_start, z_stop, r_start, r_stop, c_start, c_stop = get_bounding_box_indexes(sitk.GetArrayFromImage(liver))
        base_mask = base_mask[z_start:z_stop,r_start:r_stop,c_start:c_stop]
        pixelSpacing = np.array(label_image.GetSpacing()[::-1])
        for value in range(1,np.max(base_mask)+1):
            print(value)
            temp_dict['Group'].append(group)
            temp_dict['Patient_ID'].append('Lits_{}'.format(pat_id))
            temp_dict['Tumor_ID'].append(value)
            maskArray = base_mask == value
            SurfaceArea, Volume, diameters = cShape.calculate_coefficients(maskArray, pixelSpacing)
            sphericity = (36 * np.pi * Volume ** 2) ** (1.0 / 3.0) / SurfaceArea
            temp_dict['Sphericity'].append(sphericity)
            temp_dict['Volume_cm3'].append(Volume/10000)
            temp_dict['Voxels'].append(np.sum(maskArray))
        out_dict[pat_id] = temp_dict
        return None


def worker_def(A):
    q = A[0]
    base_class = Create_Sphericity()
    while True:
        item = q.get()
        if item is None:
            break
        else:
            base_class.process(**item)
            q.task_done()


def make_sphericity_excel(thread_count=int(cpu_count() * .9 - 1),out_path=os.path.join('.','Sphericity.xlsx')):
    q = Queue(maxsize=thread_count)
    A = [q,]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(A,))
        t.start()
        threads.append(t)
    path_base = r'H:\Liver_Disease_Ablation'
    inputs = []
    overall_dict = {}
    for folder in ['Train','Test','Validation']:
        path = os.path.join(path_base,folder)
        files = os.listdir(path)
        files = [i for i in files if i.find('Overall_mask') == 0]
        for file in files:
            inputs.append({'path':os.path.join(path,file),'out_dict':overall_dict})
    start = time.time()
    for data in inputs:
        q.put(data)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    stop = time.time()
    print('Took {} seconds'.format(stop-start))
    out_dict = {'Group':[],'Patient_ID':[], 'Tumor_ID':[],'Sphericity':[],'Volume_cm3':[],'Voxels':[]}
    for pat_id in overall_dict.keys():
        out_dict['Group'] += overall_dict[pat_id]['Group']
        out_dict['Patient_ID'] += overall_dict[pat_id]['Patient_ID']
        out_dict['Tumor_ID'] += overall_dict[pat_id]['Tumor_ID']
        out_dict['Sphericity'] += overall_dict[pat_id]['Sphericity']
        out_dict['Volume_cm3'] += overall_dict[pat_id]['Volume_cm3']
        out_dict['Voxels'] += overall_dict[pat_id]['Voxels']
    data_frame = pd.DataFrame(out_dict)
    data_frame.to_excel(out_path,index=0)
    # fid = open(os.path.join('.','password.txt'))
    # line = fid.readline()
    # fid.close()
    # data = line.split(',')
    # email_class_object = Send_Email_Class(data[0], data[1])
    # email_class_object.set_outbound_email(data[2])
    # email_class_object.send_email('test')
    return None


if __name__ == '__main__':
    make_sphericity_excel(out_path=os.path.join('.','Sphericity.xlsx')) #, thread_count=1
