import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# inp = sys.argv[1]

IP_dict = {'M04':.02, 'F03':.06, 'M12':.074, 'M01':.15, 
           'M07':.28, 'F02':.29, 'M16':.43,
           'M05':.58,  'M11':.62,'F04':.62, 
           'M09':.86,'M14':.904,  'M08':.93, 'M10':.93,'F05':.95}
# IP_dict = {'F02':.29,'F03':.06,'F04':.62, 'M04':.02,'M05':.58, 
#            'M07':.28,'M09':.86,'M10':.93,'M11':.62,'M12':.074,'M14':.904,'M16':.43}
IP_dict = dict(sorted(IP_dict.items(), key=lambda item: item[1]))
x = list(IP_dict.keys())  # Dates or categories on x-axis
y = list(IP_dict.values())
print(x)

# df = pd.read_csv(inp)
# order_df = pd.DataFrame(list(IP_dict.items()), columns=['Key', 'Order'])
# order_df.set_index('Key', inplace=True)
# df['Order'] = df['spkrs'].map(order_df['Order'])
# df = df.sort_values(by='Order').drop(columns=['Order'])

# y = [i*100 for i in y]
# df['avg_proba'] = df['avg_proba'].apply(lambda x: x * 100)
##### find correlation co-eff
# pc = pearsonr(y, df['avg_proba'] )
# print(pc)

# categories = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
# colormap = ['k', 'k', 'k', 'k', 'G', 'G', 'G', 'R', 'R', 'R', 'b', 'b', 'b', 'b', 'b']
# # colors =colormap[categories]

# plt.scatter(y[:4], df['avg_proba'][:4], s=100, c='m', marker='*')
# plt.scatter(y[4:7], df['avg_proba'][4:7], s=100, c='g', marker='*')
# plt.scatter(y[7:10], df['avg_proba'][7:10], s=100, c='b', marker='*')
# plt.scatter(y[10:], df['avg_proba'][10:], s=100, c='r', marker='*')

# pred_best12wrds = [['M01', 0.10797720840583486], ['M12', 0.0007124599885596667], ['F05', 0.9992918133333335], 
#                     ['M14', 0.8977619608333334], ['M05', 0.5651081986908333], ['F02', 0.2666768088959017],
#                     ['M16', 0.48326998342474997], ['M10', 0.9965400416666667], ['M08', 0.9830463283333333], 
#                     ['F03', 0.028830221192293847], ['M09', 0.9995149933333334], ['M04', 0.03798555545977167], 
#                     ['M11', 0.6499441818260425], ['M07', 0.21060755960349908], ['F04', 0.4880323507716034]]

pred_best12wrds =[0.03, 0.028, 0.00, 0.10, 0.21, 0.26, 0.42, 0.56, 0.649, 0.488, 0.999, 0.897, 0.98, 0.996, 0.999 ]
pred_best12wrds = [i*100 for i in pred_best12wrds]
y = [i*100 for i in y]

h = plt.scatter( pred_best12wrds[:4],y[:4], s=100, c='m', marker='*')
m = plt.scatter( pred_best12wrds[4:7], y[4:7],s=100, c='g', marker='*')
l = plt.scatter( pred_best12wrds[7:10],y[7:10], s=100, c='b', marker='*')
vl = plt.scatter( pred_best12wrds[10:], y[10:],s=100, c='r', marker='*')


###########  smiple slope plot
# plt.plot([min(y), max(y)], [df['avg_proba'].min(), df['avg_proba'].max()], 'k-')


###########  kind of linear regression plot
slope, intercept, r_value, p_value, std_err = stats.linregress(pred_best12wrds,y)

# Generate x values for the line
x_fit = np.linspace(min(pred_best12wrds), max(pred_best12wrds), 100)
# Compute y values for the line
y_fit = slope * x_fit + intercept

# print(y , df['avg_proba'])

# Plot the fitted line
plt.plot(x_fit, y_fit, color='c', linestyle='-', linewidth=2, label='Fitted Line')
 #######################        

plt.title('Scatter plot of Perceptual and predicted intelligibility score')
plt.xlabel('Predicted intelligibility score (Average probability of 12 words)')
plt.ylabel('Perceptual intelligibility score (given)')
plt.xticks(np.arange(0, 120, 20))
plt.yticks(np.arange(0, 120, 20))
plt.legend((h,m,l,vl),('high','mid','low','verylow'))
plt.show()

