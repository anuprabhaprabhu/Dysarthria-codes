import os, sys
import librosa
from tqdm import tqdm

def get_wav_duration(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=audio, sr=sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def categorize_wav_files(input_file, short_output_file, medium_output_file,large_output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    short_files = []
    medium_files = []
    large_files = []
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            wav_file_path = parts[0]
            duration = get_wav_duration(wav_file_path)
            if duration is not None:
                line_with_duration = f"{wav_file_path}\t{duration:.2f}"
                if duration < 2:
                    short_files.append(line_with_duration)
                elif 2 <= duration <= 5:
                    medium_files.append(line_with_duration)
                else:
                    large_files.append(line_with_duration)
    
    with open(short_output_file, 'w') as shortfile:
        for file in short_files:
            shortfile.write(f"{file}\n")
    
    with open(medium_output_file, 'w') as mediumfile:
        for file in medium_files:
            mediumfile.write(f"{file}\n")
    with open(large_output_file, 'w') as largefile:
        for file in large_files:
            largefile.write(f"{file}\n")

# Define the input and output file paths
input_file = '/home/anu/Desktop/UASpeech/UASpeech_noisereduce/uaspeech_1wrd/ctrl_all.txt'
short_output_file = '/home/anu/Desktop/UASpeech/UASpeech_noisereduce/uaspeech_1wrd/below_2sec_ctrl.txt'
medium_output_file = '/home/anu/Desktop/UASpeech/UASpeech_noisereduce/uaspeech_1wrd/btw_2to5_ctrl.txt'
large_output_file = '/home/anu/Desktop/UASpeech/UASpeech_noisereduce/uaspeech_1wrd/above_5sec_ctrl.txt'

print('working!!!')
# Call the function to categorize .wav files based on duration
categorize_wav_files(input_file, short_output_file, medium_output_file, large_output_file)
