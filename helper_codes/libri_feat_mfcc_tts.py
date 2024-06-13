import matplotlib.pylab as plt
from pathlib import Path
import IPython.display as ipd
import time
import sys
sys.path.append('/home2/kesavaraj.v/kesav/tacotron2/')
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
import os
import sys
import scipy.io as io
import scipy.io.wavfile as wav
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow import keras
import time
print(f'\n embed text feature extraction')
start = time.time()
hparams = create_hparams()
hparams.sampling_rate = 22050

audio_path = '/scratch/kesav/database/LibriPhrase/LibriPhrase_diffspk_all/'
audio_feat_path = '/home2/kesavaraj.v/kesav/feats/'
text_feat_path = '/home2/kesavaraj.v/kesav/feats/LibriPhrase_diffspk_all/text_embed_feats'
checkpoint_path = "/home2/kesavaraj.v/kesav/tacotron2/pretrained_models/tacotron2_statedict.pt"

model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
model = model.cuda().eval().half()

#audio feature extraction
def audio_feature_extractor(file_path):
        file = tf.io.read_file(audio_path+file_path)
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        stfts = tf.signal.stft(audio, frame_length=1024, frame_step=256, fft_length=1024)
        spectrograms = tf.abs(stfts)
        
        num_spectrogram_bins = stfts.shape[-1]
        sample_rate, lower_edge_hertz, upper_edge_hertz, num_mel_bins = 16000, 80.0, 7600.0, 80

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                                                                            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        return log_mel_spectrograms

#text feature extraction
def text_feature_extractor(text):
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))
        sequence = torch.from_numpy(sequence).unsqueeze(0).cuda()
        with torch.no_grad():
            embedding= model.embedding(sequence).transpose(1,2)
            encoder_output, conv_output = model.encoder.inference(embedding)
            embedding = embedding.squeeze(0).cpu().T

        return embedding

#concatenating all the metadata
temp=[]
csv_path = '/home2/kesavaraj.v/kesav/database/LibriPhrase/metadata'
db_list = ['train_100', 'train_360']
#forming positive samples
for db in db_list:
    for csv_file in (Path(csv_path).rglob('*' + db + '*')):
        # file_path = os.path.join(path, file)
        temp_metadata = pd.read_csv(csv_file)
        for index,row in tqdm(temp_metadata.iterrows()):
            temp.append({'audio_path':row['anchor'], 'text': row['anchor_text']})
            temp.append({'audio_path':row['comparison'], 'text': row['comparison_text']})

metadata = pd.DataFrame(temp)

#getting all unique text and audio path
unique_words = metadata['text'].unique()
unique_audio_path = metadata['audio_path'].unique()
print(len(unique_words))

 #text feature extraction
for text in tqdm(unique_words):
     file_name=text
     text_feat = text_feature_extractor(text)
     if len(text.split())>1:
         file_name = text.replace(" ","_")
         np.savez(os.path.join(text_feat_path,file_name), text_feat=text_feat)
     else:
         np.savez(os.path.join(text_feat_path,file_name), text_feat=text_feat)

end1=time.time()
duration1=(end1-start)/60
print(f"Execution time for text feature extraction is {duration1} mins")

# audio feature extraction
#for audio_filename in tqdm(unique_audio_path):
#    filename, file_extension = os.path.splitext(audio_filename)
#    audio_feat = audio_feature_extractor(audio_filename)
#    np.savez(os.path.join(audio_feat_path,filename.split("/")[-1]), audio_feat=audio_feat)

end2=time.time()
duration2=(end2-end1)/60
print(f"Execution time for audio feature extraction is {duration2} mins")

total_duration = (end2-start)/60
print(f"Execution time for total feature extraction is {total_duration} mins")
