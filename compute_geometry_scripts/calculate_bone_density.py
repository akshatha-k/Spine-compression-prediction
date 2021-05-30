import numpy as np
import nrrd
from stl import mesh
import os
import pandas as pd

rootdir = '/home/akshatha/Documents/ETH/Spine/'

df = pd.DataFrame(columns = ['patient_id', 'level', 'bone_density']) 
data = np.array(0)
mask_data = np.array(0)
for patient_id in os.listdir(rootdir): #patient_id
    patient_id_path = os.path.join(rootdir,patient_id)
    if os.path.isfile(patient_id):
        continue
    for level in os.listdir(patient_id_path): #Level
        if os.path.isfile(level):
            continue
        level_path = os.path.join(patient_id_path,level)
        for _,_,files in os.walk(level_path):
            for file in files:
                if '3.nrrd' in str(file):
                    data, header = nrrd.read(str(os.path.join(level_path, str(file))), index_order='C')
                elif '3-label.nrrd' in str(file):
                    mask_data, mask_header = nrrd.read(str(os.path.join(level_path, str(file))), index_order='C')
        bone_density = np.mean((data* mask_data))
        df = df.append({'patient_id' : patient_id, 'level' : level, 'bone_density' : bone_density},  ignore_index = True) 
df.to_csv('patient_bone_density.csv')

#         patient_id = os.path.split(os.path.dirname(subdir))[-1]
#         level = os.path.basename(subdir)
#         if '3.nrrd' in str(file):
#             data, header = nrrd.read(str(os.path.join(subdir, str(file))), index_order='C')
#             patient_data = patient_data.append({'patient_id' : patient_id, 'level' : level, 'data' : data},  ignore_index = True) 
#         if '3-label.nrrd' in str(file):
#             mask_data, mask_header = nrrd.read(str(os.path.join(subdir, str(file))), index_order='C')
#             patient_mask_data = patient_mask_data.append({'patient_id' : patient_id, 'level' : level, 'mask_data' : mask_data},  ignore_index = True) 
# print(patient_data.shape) 
# print(patient_mask_data.shape)   
# exit(0)