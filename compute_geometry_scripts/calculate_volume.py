import numpy 
from stl import mesh
import os
import pandas as pd

rootdir = '/home/akshatha/Documents/Spine'

df = pd.DataFrame(columns = ['patient_id', 'level', 'volume']) 

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if '.stl' in ext:
            patient_id = os.path.split(os.path.dirname(subdir))[-1]
            level = os.path.basename(subdir)
            #print("{} {}".format(patient_id, level))
            your_mesh = mesh.Mesh.from_file(os.path.join(subdir, file))
            volume, cog, inertia = your_mesh.get_mass_properties()
            df = df.append({'patient_id' : patient_id, 'level' : level, 'volume' : volume},  ignore_index = True) 
df.to_csv('patient_spine_volume.csv')