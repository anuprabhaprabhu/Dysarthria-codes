import numpy as np
import csv
import time
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
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
import scipy.stats
from scipy.stats import pearsonr
import cv2 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

start =time.time()


test_metadata_path = [sys.argv[1]]
# out_dir = sys.argv[2]
# test_metadata_path = ['/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/new_sil_chop/kws_ft/ft_8spkr/set0/ctrl_test.csv']
audio_test_feat_path = '/home/anuprabha/Desktop/anu_donot_touch/feats/'
text_test_feat_path = ['/home/anuprabha/Desktop/anu_donot_touch/feats/text_ua_tts/']

print("  ######################################################    ")

print(f'Taking inference for {os.path.basename(test_metadata_path[0])}')
print(test_metadata_path)


#Building character vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!. "]
char_to_num = keras.layers.StringLookup(vocabulary=characters)

#Hyperparameters
batch_size = 64
embedding_dim=256
fixed_size = (80, 200)  # (width, height)
vocab_size=char_to_num.vocabulary_size()

result=[]
for metadata_path, audio_feat_path, text_feat_path in zip(test_metadata_path, audio_test_feat_path, text_test_feat_path):
    filename, file_extension = os.path.splitext(metadata_path)
    if file_extension == ".csv":
        test_metadata = pd.read_csv(metadata_path)
        ### for generating confusion matrix
        y_test_labels = test_metadata['label']
        # print(y_test_labels)
    else:
        test_metadata = pd.read_excel(metadata_path)

    len_test = test_metadata.shape[0]                                

    def get_data_from_filename(audio, label):
        audio_file_path=audio.numpy().decode('utf-8')
        audio_file_path = os.path.join(audio_test_feat_path,audio_file_path)
        label = tf.one_hot(label, 4)
        audio_filename = audio_file_path.replace('.wav','.npz')
        audio_data = np.load(os.path.join(audio_feat_path,audio_filename))['audio_feat']
            # Resize the spectrogram
        resized_audio_data = cv2.resize(audio_data, fixed_size, interpolation=cv2.INTER_AREA)
        
        return resized_audio_data,label  

    def get_data_wrapper(file_path, text, label):
    # Assuming here that both your data and label is float type.
        audio_features,label = tf.py_function(
        get_data_from_filename, [file_path, label], (tf.float32, tf.float32))
        
        text_features = tf.strings.lower(text)
        text_features = tf.strings.unicode_split(text_features, input_encoding="UTF-8")
        text_features = char_to_num(text_features)
    
        return {'Input_1': audio_features, 'Input_2': text_features}, label

    
    padded_shapes = ({'Input_1': [200 ,80], 'Input_2': [50,]}, [4])
    # padded_shapes = ({'Input_1': [200 ,80], 'Input_2': [50,512]}, [4])
    # padded_shapes = ({'Input_1': [None ,80], 'Input_2': [50,512]}, [])

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (list(test_metadata["audio_path"]), list(test_metadata["text"]), list(test_metadata["label"]))
    )

    test_dataset =  (
        test_dataset.map(get_data_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size, padded_shapes=padded_shapes)
        .prefetch(buffer_size=tf.data.AUTOTUNE)  
    )

    loaded_model = tf.keras.models.load_model('/home/anuprabha/Desktop/anu_donot_touch/results/15spkr_classi/sd_models/mel_sd_100uncmn_char1.h5')
    # loaded_model = tf.keras.models.load_model('/home/anuprabha/Desktop/anu_donot_touch/code/models/model_ft_onlyUA_8spkr.h5',custom_objects={'f1_score':f1_score})
    # eval = loaded_model.evaluate(test_dataset)
    print(loaded_model.summary())

    feature_extractor = tf.keras.models.Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)
    
    X_test_features = feature_extractor.predict(test_dataset)
    # y_pred_prob = loaded_model.predict(test_dataset)
    

    ################# confusion matrix 
    # y_pred = np.argmax(y_pred_prob, axis=1)
    # cm = confusion_matrix(y_test_labels, y_pred)

    # # Print the confusion matrix
    # print("Confusion Matrix:")
    # print(cm)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
    #             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix')
    # plt.show()

    ################################## t-SNE plot
        # Apply t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(n_components=2, perplexity=25, learning_rate=200, n_iter=100, random_state=42)
    X_tsne = tsne.fit_transform(X_test_features)

    # Create a DataFrame for easy plotting with seaborn
    df_tsne = pd.DataFrame({
        'Dimension 1': X_tsne[:, 0],
        'Dimension 2': X_tsne[:, 1],
        'Class': y_test_labels
    })

    # Define class names for labeling
    class_names = ['Very Low', 'Low', 'Medium', 'High']
    
    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='Dimension 1', y='Dimension 2', hue='Class', palette='deep', legend='full', alpha=0.9, edgecolor=None)
    plt.title('t-SNE Visualization of 4-Class Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Class', labels=class_names)
    plt.show()
    sys.exit()

    ############################################
    
    predicted = loaded_model.predict(test_dataset)
    avg_prob = sum(predicted)/len_test
    # print(f'average of predicted probabilities ...> {sum(predicted)/len_test}')
    # out_file = f'{os.path.basename(os.path.dirname(test_metadata_path[0]))}'
    out_file = f'{os.path.basename(test_metadata_path[0])}'
    print(out_file)
 

    predicted = tf.squeeze(predicted)
    predicted = predicted.numpy()
    pred_pos = sum([i for i in predicted if i>0.5])/len_test
    out = f'/home/anuprabha/Desktop/anu_donot_touch/code/model_scripts/pred_dysar_6sec_fi/resize_spkrwisewrdproba/{out_file}'

    scores=[]
    for index,prediction in enumerate(predicted):
        scores.append({'filename': test_metadata['audio_path'][index], 'text': test_metadata['text'][index],'precition_score':prediction})

    scores_metadata = pd.DataFrame(scores)
    scores_metadata.to_csv(out, index=False)

    # with open(out,'a') as file:
    #     file.write(f'file_name={os.path.basename(test_metadata_path[0])}, acc={eval[1]},f1_score={eval[-1]},avg_prob={avg_prob[0]},positive_pred={pred_pos}\n')

print("Completed !!!!!")
