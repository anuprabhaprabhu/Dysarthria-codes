import os
import sys
from glob import glob
import pandas as pd
import re

txt_fldr = sys.argv[1]
one_wrd_fldr = sys.argv[2]


def lines_with_single_word(txt_fle):
    dest_pth = os.path.join(one_wrd_fldr, '1wrd_'+ os.path.basename(txt_fle))
    print(f'dest_pth = {dest_pth}')
    data = pd.read_csv(txt_fle, sep="\t", header=None, names=["filename", "transcript"])
    with open(dest_pth, 'a') as out_file:
        for index, row in data.iterrows():
            # wrds = row['transcript'].strip().split() 
            wrds = re.split(r'[ /\s]+', row['transcript'].strip())    

            if len(wrds) == 1:     # iclude a condition to remove 'input/images/.jpg'
                out_file.write(f"{row['filename']}\t{row['transcript']}\n")
                # print(f"{row['filename']}\t{row['transcript']}\n")

        # out_file.close()

for file in glob(os.path.join(txt_fldr,'*/*.txt')):
    print(file)
    lines_with_single_word(file)
        
print('Completed!!!')

