import pandas as pd
import numpy as np
import sys
# matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

spkrs = ['M04','M02', 'M01',  'M05', 'F01', 'F04','F03', 'M03' ]
pred_allwrds = [  0.50, 0.61, 0.44, 0.53, 0.56,  0.73,0.75, 0.91]
# pred_cmnwrds = [ 0.45, 0.47,0.54, 0.75, 0.42, 0.75,  0.80,1.0]  ### old
# pred_10wrds = [ 0.3, 0.4, 0.5, 0.3, 0.8, 0.7, 0.8,1.0]

severity = ['medium', 'medium', 'medium','low', 'low',  'very low', 'very low', 'very low']
# plt.plot(spkrs,pred_allwrds, marker='s',linestyle='-', color='g', label= ' all_wrds')
# plt.plot(spkrs,pred_cmnwrds, marker='s',linestyle='-', color='r', label= ' cmn_wrds')
ax = sns.barplot(x=spkrs, y=pred_allwrds, hue=severity, dodge=False)

plt.title('Torgo - average probability prediction (for single words)')
plt.xlabel('Dysarthria speakers (based on decreasing order of severity)')
plt.ylabel('average probability')
plt.yticks(np.arange(0,1.2,0.2))
plt.show()