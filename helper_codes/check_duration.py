import pandas as pd
from tqdm import tqdm
import librosa
import os, sys

wav_path = '/home/anuprabha/Desktop/anu_donot_touch/data'
metadata_path = '/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/clned_csv/dysar_all.csv'
metadata = pd.read_csv(metadata_path)
# new_file = '/asr3/anuprabha/anu_donot_touch/data/uaspeech/uaspeech_1wrd/train_cln.csv'
audio_paths = metadata['audio_path']

temp = []
for wav_filename in tqdm(audio_paths):
    audio_filename = os.path.join(wav_path, wav_filename)
    if os.path.exists(audio_filename):
        duration = librosa.get_duration(filename=audio_filename)
        if duration > 20:
            print(duration, wav_filename)
            temp.append({'audio_path': audio_filename, 'duration': duration})
    else:
        print(f"File not found: {audio_filename}")

print(f'files with greater than 2 sec duration is :{len(temp)}')


# max_duration_index = duration_data['duration'].idxmax()
# max_duration_audio_path = duration_data.loc[max_duration_index, 'audio_path']
# # max_duration = duration_data.loc[max_duration_index, 'duration']
# print("Maximum Duration:", max_duration)
# print("Corresponding Audio Path:", max_duration_audio_path)