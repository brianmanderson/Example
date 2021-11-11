__author__ = 'Brian M Anderson'
# Created on 3/31/2020

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join('.','Sphericity.xlsx')

df = pd.read_excel(path, engine='openpyxl')
df = df.to_dict()

out_dict = {}
for index in range(len(df['Volume_cm3'])):
    pat_id = df['Patient_ID'][index]
    volume = df['Voxels'][index]
    if pat_id in out_dict:
        out_dict[pat_id] = max([volume,out_dict[pat_id]])
    else:
        out_dict[pat_id] = volume
total_volume = []
pat_keys = np.asarray(list(out_dict.keys()))
for pat_id in pat_keys:
    total_volume.append(out_dict[pat_id])
total_volume = np.asarray(total_volume)
print(np.min(total_volume))