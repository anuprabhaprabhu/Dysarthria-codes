import os
import sys
import pandas as pd
import numpy as np
import scipy.io as io
import scipy.io.wavfile as wav
from tqdm import tqdm
from tensorflow.keras.layers import LSTM
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
from keras.layers import LeakyReLU

#defining path
train_metadata_path = '/asr3/anuprabha/anu_donot_touch/feats/uaspeech/mel_spec/train.csv'
val_metadata_path = '/asr3/anuprabha/anu_donot_touch/feats/uaspeech/mel_spec/dev.csv'
test_metadata_path = '/asr3/anuprabha/anu_donot_touch/feats/uaspeech/mel_spec/test.csv'
#audio_path = '/scratch/kesav/database/LibriPhrase/LibriPhrase_diffspk_all/'
# feat_path = '/scratch/kesav/feats/LibriPhrase_diffspk_all/audio_mfcc_feats/'

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

def get_data_from_filename(file_path, label):
   file_path=file_path.numpy().decode('utf-8')
#    print(file_path)
#    sys.exit()
#    filename = file_path.split('/')[-1].replace('.wav','.npz')
#    npdata = np.load(os.path.join(file_path, filename))
   audio_feat = np.load(file_path)['audio_feat']
  # threshold=200
  # if npdata.shape[0] < threshold:
  #     pad_width = ((0, threshold - npdata.shape[0]), (0, 0))
  #     npdata = np.pad(npdata, pad_width, mode='constant', constant_values=0)
    # Truncate if the first dimension is greater than the threshold
  # elif npdata.shape[0] > threshold:
  #     npdata = npdata[:threshold, :]
  # print(text_features.shape)
#   text_features = tf.pad(text_features, paddings, "CONSTANT")	
   
   return audio_feat, label

def get_data_wrapper(file_path, text, label):
   # Assuming here that both your data and label is float type.
    
    audio_features,label = tf.py_function(
       get_data_from_filename, [file_path, label], (tf.float32, tf.int32))
    
    text_features = tf.strings.lower(text)
    text_features = tf.strings.unicode_split(text_features, input_encoding="UTF-8")
    text_features = char_to_num(text_features)
  
    #paddings = 
    #text_features = tf.pad(text_features, paddings, "CONSTANT")

    # label = tf.one_hot(label, depth=2)
    #label = tf.cast(label, tf.float32)
   
    return {'Input_1': audio_features, 'Input_2': text_features}, label

padded_shapes = ({'Input_1': [300 ,40], 'Input_2': [50,]}, [])

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
    Input_1 = tf.keras.layers.Input(shape=(300, 40), name='Input_1')
    Input_2 = tf.keras.layers.Input(shape=(50,), name='Input_2')

    #audio model
    x = layers.Reshape((-1, 80, 1), name='Input_reshape')(Input_1)
    x = layers.Conv2D(32, 3, strides=2, activation=layers.LeakyReLU(alpha=0.2), name='convolution_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name='dropout1')(x)
    x = layers.Conv2D(64, 3, strides=1, activation=layers.LeakyReLU(alpha=0.2), name='convolution_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name='dropout2_model1')(x)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]), name='conv_reshpae')(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, activation=LeakyReLU(alpha=0.2), name='GRU_1'))(x)
    x = layers.Dropout(0.2, name='dropout3_model1')(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, activation=LeakyReLU(alpha=0.2), name='GRU_2'))(x)
    x = layers.Dropout(0.2, name='dropout4_model1')(x)
    x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2), name='output_1')(x)

    #model text
    y = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(Input_2)
    y = layers.Bidirectional(LSTM(64, return_sequences=True, activation=layers.LeakyReLU(alpha=0.2), name='First_LSTM'))(y)
    y = layers.Dropout(0.2, name='dropout1_model2')(y)
    y = layers.Bidirectional(LSTM(64, return_sequences=True, activation=layers.LeakyReLU(alpha=0.2), name='Second_LSTM'))(y)
    y = layers.Dropout(0.2, name='dropout2_model')(y)
    y = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2), name='output_2')(y)
    
    #pattern discriminator
    layer = layers.MultiHeadAttention(num_heads=2, key_dim=1, name='attention_layer')
    context_vector, attention_weights = layer(x, y, return_attention_scores=True)
    z = layers.GRU(128, return_sequences=False, activation=layers.LeakyReLU(alpha=0.2), name='GRU')(context_vector)
    z = layers.Dropout(0.2, name='dropout1_modelFinal')(z)
    z = layers.Dense(32, activation=layers.LeakyReLU(alpha=0.2))(z)
    z = layers.Dropout(0.2, name='dropout2_modelFinal')(z)
    outputs = layers.Dense(1, activation='sigmoid', name='final_layer')(z)

    model = keras.Model(inputs=[Input_1, Input_2], outputs=outputs)
    return model

with strategy.scope():    
    model = build_model()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                   loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.AUC(),
                       f1_score])
#   model = tf.keras.models.load_model('/home2/kesavaraj.v/kesav/model_scripts/baseline_sigmoid_resume/checkpoint_e20.h5',custom_objects={'f1_score':f1_score})

filepath = '/asr3/anuprabha/anu_donot_touch/model/model_mel_spec_char_embeddding'
checkpoint_every_epoch = ModelCheckpoint(filepath = filepath+'checkpoint_e{epoch:02d}.h5', period=1, save_best_only=False)
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



model.fit(train_dataset, validation_data=val_dataset, 
          workers=4,  shuffle=True, epochs=100, verbose=1)

model.evaluate(test_dataset)
