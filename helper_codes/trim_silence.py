from pydub import AudioSegment
import pandas as pd
import os, sys
from tqdm import tqdm

def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB  try with -50.0, -40.0, -30.0 
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 200 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


# ########################################################
# #### For  single file 
# sound = AudioSegment.from_file("/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/try_dur/CM01_B1_CW33_M3.wav", format="wav")

# start_trim = detect_leading_silence(sound)
# end_trim = detect_leading_silence(sound.reverse())

# duration = len(sound)   
# print(duration, start_trim, end_trim) 
# trimmed_sound = sound[start_trim:duration-end_trim]

# audio_segment = trimmed_sound.set_frame_rate(16000)
# audio_segment.export("/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/try_dur/CM01_B1_out_50.wav", format="wav")

###########################
# for csv file

wav_pth ='/home/anuprabha/Desktop/anu_donot_touch/data'
meta_pth = '/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/clned_csv/ctrl_all.csv'
new_pth = 'UASpeech/UASpeech_noisereduce/sil_trimmed'
inp = pd.read_csv(meta_pth)
# 
audio_paths = inp['audio_path']
# 
feat_info = []
# 
for index,line  in tqdm(inp.iterrows()):
    audio_filename = os.path.join(wav_pth,line['audio_path'])
    new_audio_path = os.path.join(os.path.join(wav_pth, new_pth), *audio_filename.split('/')[-3:])
# 
    new_audio_fldr = os.path.dirname(new_audio_path)
    if not os.path.exists(new_audio_fldr):
        os.makedirs(new_audio_fldr)
# 
    if os.path.exists(audio_filename):
        sound = AudioSegment.from_file(audio_filename, format="wav")
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())
        duration = len(sound)   
        trimmed_sound = sound[start_trim:duration-end_trim]
        audio_segment = trimmed_sound.set_frame_rate(16000)
        audio_segment.export(new_audio_path, format="wav")
        feat_info.append([os.path.join(new_pth, *audio_filename.split('/')[-3:]), line['text'], line['label']])    
#  
    else:
        print(f"File not found: {audio_filename}")
# 
feat_info_df = pd.DataFrame(feat_info, columns=['audio_path', 'text', 'label'])
feat_info_df.to_csv(os.path.join(wav_pth, new_pth) + '_train.csv', sep=',', index=False)
# print("Completed!!! ")