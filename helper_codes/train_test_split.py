from sklearn.model_selection import train_test_split
import pandas as pd

metadata = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/test.csv')

# train, test = train_test_split(metadata, test_size=0.1, stratify=metadata['label'])
# train, val = train_test_split(train, test_size=0.1, stratify=train['label'])

test, val = train_test_split(metadata,random_state= None ,test_size=0.1, stratify=metadata['text'])

# train.to_csv('/asr3/anuprabha/anu_donot_touch/data/uaspeech/eval_2spkr_ua/2spkr_melspec_train.csv', index=False)
val.to_csv('/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/val.csv', index=False)
test.to_csv('/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/test_new.csv', index=False)