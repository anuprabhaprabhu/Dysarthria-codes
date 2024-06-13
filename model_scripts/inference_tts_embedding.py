import numpy as np
import time
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
# import matplotlib.pyplot as plt
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
print("\n-------------------------Evaluating checkpoint 26 model in /home2/parvathipriyanka.b/kesav/model_scripts/TTS_conv_sigmoid_selfAttention------------------")
#Building character vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!. "]
char_to_num = keras.layers.StringLookup(vocabulary=characters)

#Hyperparameters
batch_size = 64
embedding_dim=256
vocab_size=char_to_num.vocabulary_size()

#test_metadata_path = ['/home2/parvathipriyanka.b/kesav/dataset_scripts/qualcomm_metadata/qualcomm_eval_metadata.csv']
#audio_test_feat_path = ['/scratch/kesav_p/feats/qualcomm/audio_mfcc_feats/']
#text_test_feat_path= ['/scratch/kesav_p/feats/qualcomm/text_conv_feats/']

test_metadata_path = ['/asr3/anuprabha/anu_donot_touch/feats/uaspeech/nosil_melspec/sil_remved_2spkr.csv']

audio_test_feat_path = ['/asr3/anuprabha/anu_donot_touch/feats/uaspeech/']

text_test_feat_path = ['/asr3/anuprabha/anu_donot_touch/feats/ua_tts/']

result=[]
for metadata_path, audio_feat_path, text_feat_path in zip(test_metadata_path, audio_test_feat_path, text_test_feat_path):
    filename, file_extension = os.path.splitext(metadata_path)
    if file_extension == ".csv":
        test_metadata = pd.read_csv(metadata_path)
    else:
        test_metadata = pd.read_excel(metadata_path)
        
    # def get_data_from_filename_1(audio,text,label):
    #     audio_file_path=audio.numpy().decode('utf-8')
    #     text = text.numpy().decode('utf-8')
    #     if len(text.split())>1:
    #         text_file_path = "_".join(text.split())
    #     else:
    #         text_file_path=text

    #     audio_filename = audio_file_path.split('/')[-1].replace('.wav','.npz')
    #     text_filename =  text_file_path+'.npz'
    #     if 'test' in metadata_path.split('/')[-1]:
    #         audio_data = np.load(os.path.join(audio_feat_path, audio_filename))['text']
    #     else:
    #         audio_data = np.load(os.path.join(audio_feat_path, audio_filename))['audio_feat']
    #     if 'test' in metadata_path.split('/')[-1]:
    #         text_data = np.load(os.path.join(text_feat_path, text_filename))['text']
    #     else:
    #         text_data = np.load(os.path.join(text_feat_path, text_filename))['text_feat']

    #     return audio_data,text_data,label

    # def get_data_wrapper_1(file_path, text, label):
    #    # Assuming here that both your data and label is float type.
  
    #     audio_features, text_features, label = tf.py_function(
    #           get_data_from_filename_1, [file_path, text, label], (tf.float32,tf.float32, tf.int32))
    
        # return {'Input_1': audio_features, 'Input_2': text_features}, label
                                    
    def get_data_from_filename(audio,text, label):
        audio_file_path=audio.numpy().decode('utf-8')
        text_file_path = text.numpy().decode('utf-8')
        audio_filename = audio_file_path.replace('.wav','.npz')
        text_filename =  text_file_path+'.npz'
        audio_data = np.load(audio_filename)['audio_feat']
        text_data = np.load(os.path.join(text_feat_path, text_filename))['text_feat']

        return audio_data, text_data, label


    def get_data_wrapper(file_path, text, label):
       # Assuming here that both your data and label is float type.
  
        audio_features, text_features, label = tf.py_function(
              get_data_from_filename, [file_path, text, label], (tf.float32,tf.float32, tf.int32))

        return {'Input_1': audio_features, 'Input_2': text_features}, label

    def f1_score(y_true, y_pred): 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    padded_shapes = ({'Input_1': [200 ,80], 'Input_2': [50,512]}, [])

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (list(test_metadata["audio_path"]), list(test_metadata["text"]), list(test_metadata["label"]))
    )
    

    test_dataset =  (
        test_dataset.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size, padded_shapes=padded_shapes)
        .prefetch(buffer_size=tf.data.AUTOTUNE)  
    )
    
    loaded_model = tf.keras.models.load_model('/asr3/anuprabha/anu_donot_touch/model/model_tts_embedding.h5',custom_objects={'f1_score':f1_score})
    loaded_model.evaluate(test_dataset)
    
    predicted = loaded_model.predict(test_dataset)
    predicted = tf.squeeze(predicted)
    predicted = predicted.numpy()
    
    # print(predicted)
    scores=[]
    for index,prediction in enumerate(predicted):
        # if prediction < 0.5:
        scores.append({'filename': test_metadata['audio_path'][index], 'precition_score':prediction})

    scores_metadata = pd.DataFrame(scores)
    scores_metadata.to_csv('prediction_2spkr_mel_spec_tts_metadata.csv', index=False)

    y_pred = np.array([1 if x >= 0.5 else 0 for x in predicted])
    y_true = np.array(test_metadata['label'])
    
    #
    #  Evaluate AUC score
    # Note: AUC score requires probability estimates, so use y_pred_probs[:, 1] for the positive class
#     auc_score = roc_auc_score(y_true, predicted)
#     print(f'AUC Score: {auc_score:.4f}')
    
#     print('classification_report')
#     report = classification_report(y_true, y_pred, output_dict=True)
#     print(classification_report(y_true, y_pred))

#     cm = confusion_matrix(y_true, y_pred)
#     print('confusion matrin\n', cm)
#     ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
#     plt.savefig(f"confusion_matrix_{filename.split('/')[-1]}.png")
#     plt.show()


#     #calculating equal error rate
    
#     fpr, tpr, thresholds = roc_curve(y_true, predicted)
   
#     #method-1 
#     eer_1 = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
#     #method-2
#     # Calculate the False Negative Rate (FNR)
#     fnr = 1 - tpr
    
#     # Calculate the absolute difference between FPR and FNR for each threshold
#     eer_values = abs(fpr - fnr)
    
#     # Find the threshold where the difference is minimized (EER)
#     eer_threshold = thresholds[eer_values.argmin()]
    
#     # The EER is the corresponding FPR and FNR at the EER threshold
#     eer = fpr[eer_values.argmin()]
    
#     print("Equal Error Rate (EER): {:.4f}" .format(eer))
#     print("Equal Error Rate_1 (EER): {:.4f}" .format(eer_1))
#     result.append({'loss':history[0], 'accuracy':history[1], 'auc_score': history[2], 'EER': eer, 'f1_score':history[3],'precision_0':report['0']['precision'],'precision_1':report['1']['precision'],'precision_macro_avg':report['macro avg']['precision'], 'recall_0':report['0']['recall'],'recall_1':report['1']['recall'],'recall_macro_avg': report['macro avg']['recall'] ,'f1_score_0':report['0']['f1-score'],'f1_score_1':report['1']['f1-score'],'f1_scrore_avg': report['macro avg']['f1-score'], 'auc_score_1': auc_score })

# print('Final_ result')
# result_df =pd.DataFrame(result)
# print(result_df.head())
# result_df.to_csv('/home2/harinie.s/kesav/model_scripts/TTS_conv_sigmoid_selfAttention/eval_metrics.csv', index=False)
# end = time.time()
# duration = (end-start)/60
# print(f"execution time is {duration} mins")
