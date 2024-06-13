import numpy as np
import time
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score as f1_sc, precision_recall_fscore_support
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import os

start =time.time()
print("Evaluating checkpoint_19 from baseline sigmoid resume")
#Building character vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!. "]
char_to_num = keras.layers.StringLookup(vocabulary=characters)

#Hyperparameters
batch_size = 64
embedding_dim=256
vocab_size=char_to_num.vocabulary_size()

test_metadata_path = ['/home2/kesavaraj.v/kesav/dataset_scripts/test_v1.csv',
                      '/home2/kesavaraj.v/kesav/dataset_scripts/libriphrase_metadata/diffspk_positive_hardneg.csv',
                      '/home2/kesavaraj.v/kesav/dataset_scripts/libriphrase_metadata/diffspk_positive_easyneg.csv',
                      '/home2/kesavaraj.v/kesav/dataset_scripts/google_metadata/google_v1_eval_metadata.csv',
                      '/home2/kesavaraj.v/kesav/dataset_scripts/qualcomm_metadata/qualcomm_eval_metadata.csv']

test_feat_path = ['/home2/kesavaraj.v/kesav/feats/LibriPhrase_diffspk_all/audio_mfcc_feats/',
                  '/home2/kesavaraj.v/kesav/feats/train-other-500/audio_mfcc_feats/',
                  '/home2/kesavaraj.v/kesav/feats/train-other-500/audio_mfcc_feats/',
                  '/home2/kesavaraj.v/kesav/feats/google_v1/audio_mfcc_feats/',
                  '/home2/kesavaraj.v/kesav/feats/qualcomm/audio_mfcc_feats/']

for metadata_path, feat_path in zip(test_metadata_path, test_feat_path):
    filename, file_extension = os.path.splitext(metadata_path)
    if file_extension == ".csv":
        test_metadata = pd.read_csv(metadata_path)
    else:
        test_metadata = pd.read_excel(metadata_path)
        
    def get_data_from_filename_1(file_path, label):
        file_path=file_path.numpy().decode('utf-8')
        filename = file_path.split('/')[-1].replace('.wav','.npz')
        npdata = np.load(os.path.join(feat_path, filename))
        if 'test' in metadata_path.split('/')[-1]:
             npdata=npdata['text']
        else:
             npdata=npdata['audio_feat']
        return npdata, label

    def get_data_wrapper_1(file_path, text, label):
       # Assuming here that both your data and label is float type.
  
        audio_features,label = tf.py_function(
           get_data_from_filename_1, [file_path, label], (tf.float32, tf.int32))

        text_features = tf.strings.lower(text)
        text_features = tf.strings.unicode_split(text_features, input_encoding="UTF-8")
        text_features = char_to_num(text_features)
    
        return {'Input_1': audio_features, 'Input_2': text_features}, label
                                    
    def get_data_from_filename(file_path, label):
       file_path=file_path.numpy().decode('utf-8')
       filename = file_path.replace('.wav','.npz')
       npdata = np.load(os.path.join(feat_path, filename))
       npdata=npdata['audio_feat']

       return npdata, label


    def get_data_wrapper(file_path, text, label):
       # Assuming here that both your data and label is float type.
  
        audio_features,label = tf.py_function(
           get_data_from_filename, [file_path, label], (tf.float32, tf.int32))

        text_features = tf.strings.lower(text)
        text_features = tf.strings.unicode_split(text_features, input_encoding="UTF-8")
        text_features = char_to_num(text_features)

        return {'Input_1': audio_features, 'Input_2': text_features}, label

    def f1_score(y_true, y_pred): 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    padded_shapes = ({'Input_1': [200 ,80], 'Input_2': [50]}, [])

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (list(test_metadata["audio_path"]), list(test_metadata["text"]), list(test_metadata["label"]))
    )
    
    if 'hardneg' in metadata_path.split('/')[-1] or 'easyneg' in metadata_path.split('/')[-1] or 'test' in metadata_path.split('/')[-1]:
        test_dataset =  (
            test_dataset.map(get_data_wrapper_1, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size, padded_shapes=padded_shapes)
             .prefetch(buffer_size=tf.data.AUTOTUNE)  
        )
    else:
        test_dataset =  (
            test_dataset.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
            .padded_batch(batch_size, padded_shapes=padded_shapes)
            .prefetch(buffer_size=tf.data.AUTOTUNE)  
        )
    
    print(f"\n---------Evaluating {filename}-----------")
    loaded_model = tf.keras.models.load_model('/home2/kesavaraj.v/kesav/model_scripts/baseline_sigmoid_v1_v2/checkpoint_e09.h5',custom_objects={'f1_score':f1_score})
    loaded_model.evaluate(test_dataset)
    
    predicted = loaded_model.predict(test_dataset)
    predicted = tf.squeeze(predicted)
    
    y_pred = np.array([1 if x >= 0.5 else 0 for x in predicted])
    y_true = np.array(test_metadata['label'])
    
    #
    #  Evaluate AUC score
    # Note: AUC score requires probability estimates, so use y_pred_probs[:, 1] for the positive class
    auc_score = roc_auc_score(y_true, predicted)
    print(f'AUC Score: {auc_score:.4f}')
    
    print('classification_report')
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print('confusion matrin\n', cm)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(f"confusion_matrix_{filename.split('/')[-1]}.png")
    plt.show()


    #calculating equal error rate
    
    fpr, tpr, thresholds = roc_curve(y_true, predicted)
   
    #method-1 
    eer_1 = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    #method-2
    # Calculate the False Negative Rate (FNR)
    fnr = 1 - tpr
    
    # Calculate the absolute difference between FPR and FNR for each threshold
    eer_values = abs(fpr - fnr)
    
    # Find the threshold where the difference is minimized (EER)
    eer_threshold = thresholds[eer_values.argmin()]
    
    # The EER is the corresponding FPR and FNR at the EER threshold
    eer = fpr[eer_values.argmin()]
    
    print("Equal Error Rate (EER): {:.4f}" .format(eer))
    print("Equal Error Rate_1 (EER): {:.4f}" .format(eer_1))

end = time.time()
duration = (end-start)/60
print(f"execution time is {duration} mins")
