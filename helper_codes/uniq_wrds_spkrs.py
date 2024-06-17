import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd


file1 = sys.argv[1]
# file2 = sys.argv[2]

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
    # for i in uni_wrds:
    #     if len(i)==1:
    #         print(i)
    # sys.exit()
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

    print( f" /n Out of Vocabulary words in file2: {len(oov_words_file2)} ")
    # print(oov_words_file2)
    return common_words, oov_words_file1, oov_words_file2

def uniq_spkrs(file1):
    with open(file1, 'r') as file:
        lines = file.readlines()

    spkrs = []

    for line in lines:
        path, label, _ = line.strip().split(',')
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
        path, text, _ = line.strip().split(',')
        filename = os.path.basename(path)
        speaker_name = filename.split('_')[0]
        words = text.split()
        speaker_words[speaker_name].update(words)

    print(f'For file {os.path.basename(file1)} :', '\n')
    for speaker, words in speaker_words.items():
        print(f'{speaker}: {len(words)} unique words')

    
def copy_speaker_contents(file):
    spkrs = ['CF04','CF05','CM12','CM13']
    df = pd.read_csv(file , sep='/t', names=['audio_path', 'text', 'label'])
    filtered_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(spkrs)]
    print(filtered_df.head())
    filtered_df.to_csv(os.path.join(os.path.dirname(file),'ctrl_ft_4spkr_set2.csv'), sep=',', header=False, index=False)
    print("Completed !!!")


def block_contents(file):
    spkrs = ['B1']
    # spkrs = ['M3']

    df = pd.read_csv(file , sep=',', names=['audio_path', 'text', 'label'])
    # filtered_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('.')[0].split('_')[-1]).isin(spkrs)]
    filtered_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[1]).isin(spkrs)]
    out = os.path.join(os.path.dirname(file),'ctrl_B1.csv')
    filtered_df.to_csv(out, index=False)
    print("Completed !!!")
    return out

def wrd_spkr_distri(file):
    word_count = defaultdict(lambda: defaultdict(int))
    print(file)
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        spkr = row['audio_path'].split('/')[-2]
        word = row['text']
        word_count[spkr][word] += 1

    # Create a new dictionary for word-centric view
    word_speaker_count = defaultdict(lambda: defaultdict(int))

    # Populate the new dictionary
    for speaker, words in word_count.items():
        for word, count in words.items():
            word_speaker_count[word][speaker] += count

    # Convert the word_speaker_count dictionary to a sorted list of tuples
    sorted_word_speaker_count = sorted(word_speaker_count.items(), key=lambda item: sum(item[1].values()), reverse=True)

    # Print the header
    print(f"{'Word':<15} {'Total Count':<12} {'Speakers':<40}")
    print("="*70)

    # Print the sorted word counts and speakers
    for word, speakers in sorted_word_speaker_count:
        total_count = sum(speakers.values())
        speaker_counts = ', '.join([f"{speaker}: {count}" for speaker, count in speakers.items()])
        print(f"{word:<15} {total_count:<12} {speaker_counts:<40}")


##########################################


# common_words, oov_words_file1, oov_words_file2 = find_common_and_oov_words(file1, file2)
# # copy_speaker_contents(file1)
# file1 = block_contents(file1)
# uniq_words(file1)
# unique_words_per_speaker(file1)
# uniq_spkrs(file1)
wrd_spkr_distri(file1)