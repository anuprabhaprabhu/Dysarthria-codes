import os,sys
from my_snr import value_snr
import pandas as pd
from tqdm import tqdm

wav_pth ='/home/anuprabha/Desktop/anu_donot_touch/data'
meta_pth = '/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/snr_chk.csv'

inp = pd.read_csv(meta_pth)

for index,line  in tqdm(inp.iterrows()):
    audio_filename = os.path.join(wav_pth,line['audio_path'])
    snr_value = value_snr(audio_filename)
    print(snr_value, os.path.basename(audio_filename))
    # sys.exit()

print("Completed!!!!!")


