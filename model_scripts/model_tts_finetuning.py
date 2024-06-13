import os
import sys
import pandas as pd
import numpy as np
import scipy.io as io
import scipy.io.wavfile as wav
from tqdm import tqdm
from tensorflow.keras.layers import LSTM, GRU
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import multiprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow import keras

#defining path
train_metadata_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/ft_ctrl_4spkr/sil_trimmed/melspec/train_feats.csv'
val_metadata_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/ft_ctrl_4spkr/sil_trimmed/melspec/val_feats.csv'
test_metadata_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/ft_ctrl_4spkr/sil_trimmed/melspec/test_feats.csv'
audio_feat_path = '/home/anuprabha/Desktop/anu_donot_touch/feats'
text_feat_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/text_ua_tts'

train_metadata = pd.read_csv(train_metadata_path)
val_metadata = pd.read_csv(val_metadata_path)
test_metadata = pd.read_csv(test_metadata_path)

#Building character vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!. "]
char_to_num = keras.layers.StringLookup(vocabulary=characters)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

#Hyperparameters
batch_size = 64
embedding_dim=256
vocab_size=char_to_num.vocabulary_size()


strategy = tf.distribute.MirroredStrategy()
batch_size = batch_size * strategy.num_replicas_in_sync

def get_data_from_filename(audio, text, label):
   audio_file_path=audio.numpy().decode('utf-8')
   text = text.numpy().decode('utf-8')
   text_file_path = text.replace(" ", "_")
   audio_filename = audio_file_path.replace('.wav','.npz')
   text_filename =  text_file_path+'.npz'
   audio_data = np.load(os.path.join(audio_feat_path,audio_filename))['audio_feat']
   text_data = np.load(os.path.join(text_feat_path, text_filename))['text_feat']

   return audio_data,text_data,label

def get_data_wrapper(file_path, text, label):
   # Assuming here that both your data and label is float type.
    print(file_path, text)
    audio_features, text_features, label = tf.py_function(
       get_data_from_filename, [file_path, text, label], (tf.float32,tf.float32, tf.int32))
    return {'Input_1': audio_features, 'Input_2': text_features}, label

padded_shapes = ({'Input_1': [200 ,80], 'Input_2': [50, 512]}, [])

train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(train_metadata["audio_path"]), list(train_metadata["text"]), list(train_metadata["label"]))
)

train_dataset =  (
    train_dataset.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size, padded_shapes=padded_shapes)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (list(val_metadata["audio_path"]), list(val_metadata["text"]), list(val_metadata["label"]))
)

val_dataset =  (
    val_dataset.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size, padded_shapes=padded_shapes)
    .prefetch(buffer_size=tf.data.AUTOTUNE)  
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (list(test_metadata["audio_path"]), list(test_metadata["text"]), list(test_metadata["label"]))
)

test_dataset =  (
    test_dataset.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size, padded_shapes=padded_shapes)
    .prefetch(buffer_size=tf.data.AUTOTUNE)  
)

def build_model():
    Input_1 = tf.keras.layers.Input(shape=(200, 80), name='Input_1')
    Input_2 = tf.keras.layers.Input(shape=(50, 512), name='Input_2')

    #audio model
    x = layers.Reshape((-1, 80, 1), name='Input_reshape')(Input_1)
    x = layers.Conv2D(32, 3, strides=2, activation=layers.LeakyReLU(alpha=0.2), name='convolution_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name='dropout1')(x)
    x = layers.Conv2D(64, 3, strides=1, activation=layers.LeakyReLU(alpha=0.2), name='convolution_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name='dropout2_model1')(x)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]), name='conv_reshpae')(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, activation=layers.LeakyReLU(alpha=0.2), name='GRU_1'))(x)
    x = layers.Dropout(0.2, name='dropout3_model1')(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, activation=layers.LeakyReLU(alpha=0.2), name='GRU_2'))(x)
    x = layers.Dropout(0.2, name='dropout4_model1')(x)
    x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2), name='output_1')(x)

    #model text
    y = layers.Bidirectional(GRU(64, return_sequences=True, activation=layers.LeakyReLU(alpha=0.2), name='First_LSTM'))(Input_2)
    y = layers.Dropout(0.2, name='dropout1_model2')(y)
   # y = layers.Bidirectional(LSTM(64, return_sequences=True, activation=layers.LeakyReLU(alpha=0.2), name='Second_LSTM'))(y)
   # y = layers.Dropout(0.2, name='dropout2_model')(y)
    y = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2), name='output_2')(y)
    
    #pattern discriminator
    layer = layers.MultiHeadAttention(num_heads=2, key_dim=1, name='attention_layer')
    context_vector, attention_weights = layer(y, x, return_attention_scores=True)
    z = layers.GRU(128, return_sequences=False, activation=layers.LeakyReLU(alpha=0.2), name='GRU')(context_vector)
    z = layers.Dropout(0.2, name='dropout1_modelFinal')(z)
    z = layers.Dense(32, activation=layers.LeakyReLU(alpha=0.2))(z)
    z = layers.Dropout(0.2, name='dropout2_modelFinal')(z)
    outputs = layers.Dense(1, activation='sigmoid', name='final_layer')(z)

    model = keras.Model(inputs=[Input_1, Input_2], outputs=outputs)
    return model

with strategy.scope():     
    #model = build_model()

    #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #                loss=tf.keras.losses.BinaryCrossentropy(),
    #                metrics=[tf.keras.metrics.BinaryAccuracy(),
    #                        tf.keras.metrics.AUC(),
    #                        f1_score])
    model = tf.keras.models.load_model('/home/anuprabha/Desktop/anu_donot_touch/code/models/model_mel_lstm.h5',custom_objects={'f1_score':f1_score})

filepath = '/home/anuprabha/Desktop/anu_donot_touch/code/models/model_tts_finetuning/ft_ctrl_4spkrs/sil_trimmed/melspec'
if not os.path.exists(filepath):
    os.makedirs(filepath)

checkpoint_every_epoch = ModelCheckpoint(filepath = filepath+'checkpoint_e{epoch:02d}.h5', period=2, save_best_only=False)
checkpoint_best = ModelCheckpoint(filepath = filepath+'best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2)



model.fit(train_dataset, validation_data=val_dataset, workers=4,  
          shuffle=True, epochs=10, verbose=1)

model.evaluate(test_dataset)
