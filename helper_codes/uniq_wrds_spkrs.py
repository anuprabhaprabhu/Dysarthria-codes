import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd


file1 = sys.argv[1]
file2 = sys.argv[2]
# file3 = sys.argv[3]

def uniq_words(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    wrds = []
    for line in lines:
        # path, text, _ = line.strip().split('\t')  #for tab seperated file 
        path, text, _ = line.strip().split(',')   # for comma seperated files
        wrd = os.path.basename(text)
        wrds.append(wrd)
    uni_wrds = set(wrds)
    print(f'Unique words present in the file {file_path} is {len(uni_wrds)}')
    return uni_wrds

def find_common_and_oov_words(file1, file2):
    words1 = uniq_words(file1)
    words2 = uniq_words(file2)
    
    # common_words = words1 & words2
    common_words = words1.intersection(words2)
    oov_words_file1 = words1 - words2
    oov_words_file2 = words2 - words1
    print(f"Number of common words: {len(common_words)}. Common words:")
    # print(common_words )

    print(f"Out of Vocabulary words in file1:{len(oov_words_file1)}")

    print( f" /n OOV words in test set {len(oov_words_file2)} /nOut of Vocabulary words in file2:")
    # print(oov_words_file2)
    return common_words, oov_words_file1, oov_words_file2

def uniq_spkrs(file1):
    with open(file1, 'r') as file:
        lines = file.readlines()

    spkrs = []

    for line in lines:
        path, label, _ = line.strip().split('\t')
        filename = os.path.basename(path)
        spkr_name = filename.split('_')[0]  
        spkrs.append(spkr_name)

    unique_spkrs = set(spkrs)
    num_unique_spkrs = len(unique_spkrs)
    print(f'For file {os.path.basename(file1)} : ', '\n')
    print(f'Number of unique speakers= {num_unique_spkrs} and  {set(spkrs)}')

    spkr_counts = Counter(spkrs)
    for spkr, count in spkr_counts.items():
        print(f'{spkr}: {count}')

def unique_words_per_speaker(file1):
    with open(file1, 'r') as file:
        lines = file.readlines()

    speaker_words = defaultdict(set)

    for line in lines:
        path, text, _ = line.strip().split('\t')
        filename = os.path.basename(path)
        speaker_name = filename.split('_')[0]
        words = text.split()
        speaker_words[speaker_name].update(words)

    print(f'For file {os.path.basename(file1)} :', '\n')
    for speaker, words in speaker_words.items():
        print(f'{speaker}: {len(words)} unique words')

    
def copy_speaker_contents(file1):
    spkrs = ['CF02','CF03','CM04','CM05']
    df = pd.read_csv(file1, sep='\t', names=['Path', 'Label', 'Utterance'])
    filtered_df = df[df['Path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkrs)]
    filtered_df.to_csv(os.path.join(os.path.dirname(file1),'ctrl_ft_4spkr.txt'), sep='\t', header=False, index=False)
    print("Completed !!!")


common_words, oov_words_file1, oov_words_file2 = find_common_and_oov_words(file1, file2)


# uniq_words(file1)
# unique_words_per_speaker(file1)
# uniq_spkrs(file1)
# copy_speaker_contents(file1)