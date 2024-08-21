import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

inp_csv = sys.argv[1]
df = pd.read_csv(inp_csv)

# print(df.head(5))

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 10))
axes = axes.flatten()

classes = df['groups'].unique()
print(classes)
colors = ['b', 'g', 'r', 'k']  

for i, cls in enumerate(classes):
    df_class = df[df['groups'] == cls]
    # print(df_class.head())
    # sys.exit()
    subclass = df_class['class'].unique()
    

    for j, sub in enumerate(subclass):
        print(sub)
        df_subclass = df_class[df_class['class'] == sub]
        # print(df_subclass.head(5))
        df_subclass['mean_acc'] = df_subclass['acc'].mean()
        # print(mean_acc)
        # print(df_subclass.head(5))
        # sys.exit()

        axes[i].scatter(df_subclass['class'], df_subclass['avg_proba'], color=colors[j], 
                        alpha=0.7) #, label=f'{sub}')
    
    axes[i].set_title(f'{cls}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    # axes[i].legend(title='Subclass')
# fig.legend(handles, labels, loc='upper center', ncol=len(colors), title='Subclass')
plt.tight_layout()
plt.show()
