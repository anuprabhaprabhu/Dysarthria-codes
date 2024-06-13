import os, sys
from tqdm import tqdm
import numpy as np
import pandas as pd


feat_pth ='/home/anuprabha/Desktop/anu_donot_touch/feats'
metadata_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/ft_ctrl_4spkr/sil_trimmed/melspec/val_feats.csv'

metadata = pd.read_csv(metadata_path)

# feat_info = []

# for index,line  in tqdm(metadata.iterrows()):
#     feat_filename = os.path.join(feat_pth,line['audio_path'])
#     data = np.load(feat_filename)

#     for key in data.keys():
#         print(f"Array name: {line['audio_path']}, Shape: {data[key].shape}")
#     # sys.exit()
# print("Completed!!! ")

# Target shape criteria
target_shape = (200, 80)

# Initialize a list to track the file names meeting the shape criteria
matching_files = []
max_shape =[]
for index, line in tqdm(metadata.iterrows(), total=metadata.shape[0]):
    feat_filename = os.path.join(feat_pth, line['audio_path'])
    
    if os.path.exists(feat_filename):
        data = np.load(feat_filename)
        
        for key in data.keys():
            shape = data[key].shape
            # matching_files.append(shape)
            if shape[0] > 200 : #target_shape[0]: # and shape[1] > target_shape[1]:
                matching_files.append(line['audio_path'])
                max_shape.append(shape[0])  
                print(os.path.basename(line['audio_path']))             
                break  # Stop checking other arrays in this file once a match is found
    else:
        print(f"File not found: {feat_filename}")
print(max_shape)
# print(f"Files with shapes greater than {target_shape}: {matching_files}")
# print(f"Files with shapes greater than {target_shape}: {len(matching_files)}")
print(f'max value of shape: {max(max_shape)}')
print(f'min value of shape: {min(max_shape)}')
