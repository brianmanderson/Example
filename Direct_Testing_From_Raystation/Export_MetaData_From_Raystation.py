__author__ = 'Brian M Anderson'
# Created on 5/26/2020
import os

path = r'H:\Liver_Disease_Ablation\test_export'
exam_name = 'CT 16'
case.Examinations[exam_name].ExportExaminationAsMetaImage(MetaFileName=os.path.join(path,'Examination.mhd'))
case.PatientModel.StructureSets[exam_name].RoiGeometries['Liver'].ExportRoiGeometryAsMetaImage(MetaFileName=os.path.join(path,'Liver.mhd'),AsExamination=True)
case.PatientModel.StructureSets[exam_name].RoiGeometries['Ablation'].ExportRoiGeometryAsMetaImage(MetaFileName=os.path.join(path,'Ablation.mhd'),AsExamination=True)