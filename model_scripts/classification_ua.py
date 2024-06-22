import os
import sys
import pandas as pd
import numpy as np
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

inp = sys.argv[1]
epoch = int(sys.argv[2])
pad_feat = int(sys.argv[3])

#defining path
train_metadata_path = f'/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/classification/{inp}/train_{inp}.csv'
val_metadata_path = f'/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/classification/{inp}/val_{inp}.csv'
test_metadata_path = f'/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/classification/{inp}/test_{inp}.csv'
audio_feat_path = '/home/anuprabha/Desktop/anu_donot_touch/feats'
text_feat_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/text_ua_tts'

# print(train_metadata_path)
# print(val_metadata_path)
# print(test_metadata_path)
# print(f'/home/anuprabha/Desktop/anu_donot_touch/model/classificatio_ua/set1/{inp}')
# print({'Input_1': [200 ,pad_feat], 'Input_2': [50, 512]}, [])
# print(f'/home/anuprabha/Desktop/anu_donot_touch/code/models/model_{inp}_lstm.h5')
# print(epoch)



train_metadata = pd.read_csv(train_metadata_path)
val_metadata = pd.read_csv(val_metadata_path)
test_metadata = pd.read_csv(test_metadata_path)


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

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

#Hyperparameters
batch_size = 64

strategy = tf.distribute.MirroredStrategy()
batch_size = batch_size * strategy.num_replicas_in_sync

padded_shapes = ({'Input_1': [200 ,pad_feat], 'Input_2': [50, 512]}, [])

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


with strategy.scope():     
    base_model = tf.keras.models.load_model(f'/home/anuprabha/Desktop/anu_donot_touch/code/models/model_{inp}_lstm.h5',custom_objects={'f1_score':f1_score})
    # # Freeze layers
    # for layer in base_model.layers[:-5]:
    #     layer.trainable = False
    
    base_model.summary()
    # # # take output from the last frozen layer
    # context_vector, attention_weights = base_model.layers[-6].output
    # # # custom layers 
    # z = layers.GRU(128, return_sequences=False, activation=layers.LeakyReLU(alpha=0.2), name='custom_GRU')(context_vector)
    # z = layers.Dropout(0.2, name='custom_dropout1_modelFinal')(z)
    # z = layers.Dense(32, activation=layers.LeakyReLU(alpha=0.2))(z)
    # z = layers.Dropout(0.2, name='custom_dropout2_modelFinal')(z)
    # out = layers.Dense(1, activation='sigmoid', name='custom_final_layer')(z)
    # # x = layers.Dense(16, activation=layers.LeakyReLU(alpha=0.2), name='Custom_dense')(x)
    # # x = layers.Dropout(0.2, name='custom_dropout')(x)
    # # out = layers.Dense(1, activation='sigmoid', name='final')(x)
    # model = Model(inputs=base_model.input, outputs=out)

    # model.summary()          
    print("Hello !!!!!!!")
    sys.exit()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                   loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC(),
                           f1_score])
 
filepath = f'/home/anuprabha/Desktop/anu_donot_touch/model/classificatio_ua/set1/{inp}/'
if not os.path.exists(filepath):
    os.makedirs(filepath)

checkpoint_every_epoch = ModelCheckpoint(filepath = filepath+'checkpoint_e{epoch:02d}.h5', period=5, save_best_only=False)
checkpoint_best = ModelCheckpoint(filepath = filepath+'best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2)


# orig_stdout = sys.stdout
# f = open(f'/home/anuprabha/Desktop/anu_donot_touch/model/classificatio_ua/set1/train_log_{inp}.txt', 'w')
# sys.stdout = f

model.fit(train_dataset, validation_data=val_dataset, 
          workers=4, shuffle=True, epochs=epoch, verbose=1,  
          callbacks=[checkpoint_every_epoch, checkpoint_best, early_stopping, lr_sched])

model.evaluate(test_dataset)

# sys.stdout = orig_stdout
# f.close()

print("Completed !!!!")