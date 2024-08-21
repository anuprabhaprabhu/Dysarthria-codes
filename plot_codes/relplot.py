import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

inp_csv = sys.argv[1]

df = pd.read_csv(inp_csv)

sns.set_palette("bright")
sns.relplot(data=df, x="groups", y="acc", hue="class", marker='H')
plt.title('Fig 1. Average probability of each speaker for different word groups. D-digits, C-computer commands,\
            L - letters, CW - common words, UW - uncommon words, B1, B2, B3 - each blocks')
# plt.legend(title='')
plt.xlabel('Word groups')
plt.ylabel('average probability')
plt.show()