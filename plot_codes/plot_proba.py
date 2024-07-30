import pandas as pd
import numpy as np
import sys
# matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import re 
from scipy.stats import norm

inp = sys.argv[1]

# df = pd.read_csv(inp, sep=",", header=None)
# df= pd.read_csv(inp, sep=",", header=None, names= ['speakers','probability'])
# df['speakers'] = df['speakers'].str.replace('.csv','')
# df['speakers'] = df['speakers'].str.replace('filename:','')
# df['probability'] = df['probability'].str.replace('probabilites :','')
# df['probability'] = df['probability'].map(lambda x: float(x))


def process_text_file(input_file):
    # Read the content of the text file
    with open(input_file, 'r') as file:
        input_string = file.read()
    
    # Extract all filenames and probabilities
    filename_pattern = re.compile(r'filename:(\S+), probabilites :(\[\[.*?\]\])')
    matches = filename_pattern.findall(input_string)

    data = []
    for filename, probabilities_str in matches:
        # Clean up the probabilities string
        probabilities_str = probabilities_str.strip('[]')
        
        # Convert the cleaned string to a list of floats
        probabilities_str = probabilities_str.replace('], [', ',').replace('[', '').replace(']', '')
        probabilities_list = [float(x) for x in probabilities_str.split(',')]
        
        # Append data for each probability
        for prob in probabilities_list:
            data.append({'Filename': filename, 'Probability': prob})

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    return df
# Process the text file
df = process_text_file(inp)
df['Filename'] = df['Filename'].str.replace('.csv','')
print(df.head())

filename = 'M14'
df_filtered = df[df['Filename'] == filename]
# Extract the probabilities
probabilities = df_filtered['Probability']

# Fit a normal distribution to the data
mean, std = norm.fit(probabilities)

# Plot histogram and normal distribution
plt.figure(figsize=(10, 6))
sns.histplot(probabilities, bins=20, kde=False, color='blue', stat='density', label='Histogram')

# Overlay a normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')

plt.title(f'Histogram and Normal Distribution Fit for {filename}')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend()
plt.show()
# # Get unique filenames
# filenames = df['Filename'].unique()
# print(filenames)
# # Set up the subplots
# num_files = len(filenames)
# fig, axes = plt.subplots(num_files, 1, figsize=(10, 5 * num_files), sharex=True)

# # If there is only one subplot, axes is not an array
# if num_files == 1:
#     axes = [axes]

# # Plot histograms for each filename
# for ax, filename in zip(axes, filenames):
#     df_filtered = df[df['Filename'] == filename]
#     sns.histplot(df_filtered['Probability'], bins=20, kde=True, color='blue', ax=ax)
#     ax.set_title(f'Histogram of Probabilities for {filename}')
#     ax.set_xlabel('Probability')
#     ax.set_ylabel('Frequency')
#     break

# plt.tight_layout()
plt.show()




# plt.show()

