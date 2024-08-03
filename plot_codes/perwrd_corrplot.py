import pandas as pd
import numpy as np
import os, sys
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy import stats

fldr = sys.argv[1]

IP_dict = {'M04':.02, 'F03':.06, 'M12':.074, 'M01':.15, 'M07':.28, 'F02':.29, 'M16':.43,'M05':.58, 
        'M11':.62,'F04':.62, 'M09':.86,'M14':.904,  'M08':.93, 'M10':.93,'F05':.95}

#################################################

files = [i for i in os.listdir(fldr) if i.endswith('.csv')]
# print(files)
# df_togetwrds = pd.read_csv('/home/anuprabha/Desktop/anu_donot_touch/feats/uaspeech/new_sil_chop/kws_ft/assessment/dysar/spkr_wise/M10.csv')
# all_wrds = df_togetwrds['text']

# print(len(all_wrds))
common_set = ['stringy','naturalization','advantageous', 'gigantic','footmarks', 'moustache', 'moisten','cheshire','ashamed','go','cup','about']
# common_set = ['cup','cheshire']

# common_set = ['advantageous','astounded' ,'moustache','yes','sugar','skyward','do','interwoven','so','that'
#             ,'battlefield','irresolute','in','she','stringy','shovel','to','come','moisten','hypothesis','merchandise']
#  [0.87,0.86, 0.84,0.83.0.83,0.83,0.81,0.80,0.80,0.80,0.79,0.79,0.78,0.78,0.78,0.77,0.77,0.77,0.77,0.77 ]
# print(common_set)
wrd_pred = []

for i in files:
    df = pd.read_csv(os.path.join(fldr,i))
    wrd_df = df[df['text'].isin(common_set)]
    wrd_df = wrd_df.drop_duplicates(subset=['text'])
    sum_prob = wrd_df['precition_score'].sum()
    # print(type(sum_prob))
    sum_prob = sum_prob/len(common_set)  # [i/len(common_set) for i in sum_prob]
    wrd_pred.append([i.replace('.csv',''), sum_prob])
# print(wrd_pred)
#######################  find  Pearson correlation ###########
x =[]
y= []
x_vl= []
y_vl =[]
x_l =[]
y_l =[]
x_m =[]
y_m =[]
x_h = []
y_h = []

for idx, s in enumerate(wrd_pred):
    x.append(IP_dict[s[0]]*100)
    y.append(s[1]*100)
    if s[0] in ['F03','M01','M04','M12']:
        x_vl.append(IP_dict[s[0]]*100)
        y_vl.append(s[1]*100)
    if s[0] in ['F02','M07','M16']:
        x_l.append(IP_dict[s[0]]*100)
        y_l.append(s[1]*100)
    if s[0] in ['F04','M05','M11']:
        x_m.append(IP_dict[s[0]]*100)
        y_m.append(s[1]*100)
    if s[0] in ['F05','M08','M09','M10','M14']:
        x_h.append(IP_dict[s[0]]*100)
        y_h.append(s[1]*100)

    # print(s[0], IP_dict[s[0]], s[1])
pc = pearsonr( y, x)


print(pc)
print(wrd_pred)
h = plt.scatter( y_h,x_h, s=100, c='m', marker='*')
m = plt.scatter( y_m,x_m,s=100, c='g', marker='*')
l = plt.scatter( y_l,x_l, s=100, c='b', marker='*')
vl = plt.scatter( y_vl,x_vl,s=100, c='r', marker='*')
plt.xlabel('Predicted intelligibility score')
plt.ylabel('Perceptual intelligibility score (given)')
plt.xticks(np.arange(0, 120, 20))
plt.yticks(np.arange(0, 120, 20))
plt.legend((h,m,l,vl),('high','mid','low','verylow'))

###########  kind of linear regression plot
slope, intercept, r_value, p_value, std_err = stats.linregress(y,x)

# Generate x values for the line
x_fit = np.linspace(min(y), max(y), 100)
# Compute y values for the line
y_fit = slope * x_fit + intercept

# print(y , df['avg_proba'])

# Plot the fitted line
plt.plot(x_fit, y_fit, color='c', linestyle='-', linewidth=2, label='Fitted Line')


plt.show()
print("Completted !!!!!")