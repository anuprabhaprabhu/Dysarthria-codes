import pandas as pd
import numpy as np

# Load the CSV file

df = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/test.csv')

# Extract all unique words from the 'text' column
unique_words = set()
df['text'].str.split().apply(unique_words.update)


# Split the unique words into two sets of 125 words each
unique_words = list(unique_words)
np.random.shuffle(unique_words)  # Shuffle to ensure randomness
test_words = set(unique_words[:125])
dev_words = set(unique_words[125:])

# Function to check if a row contains any word from a given set
def contains_any(row, word_set):
    words = set(row.split())
    return not words.isdisjoint(word_set)

# Split the DataFrame into test and dev sets based on the unique words
test_df = df[df['text'].apply(contains_any, args=(test_words,))]
dev_df = df[df['text'].apply(contains_any, args=(dev_words,))]

# Save the test and dev DataFrames to new CSV files
dev_df.to_csv('/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/val.csv', index=False)
test_df.to_csv('/home/anuprabha/Desktop/anu_donot_touch/data/UASpeech/UASpeech_noisereduce/FT_ctrl_4spkr/test_new.csv', index=False)

print("Test and development files created successfully.")