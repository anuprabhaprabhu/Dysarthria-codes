import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

spkrs = ['M12','M04', 'F03',  'M01',   'M07', 'F02', 'M16',   'F04', 'M11', 'M05',  'M14', 'M08', 'M10', 'F05', 'M09']

###  spkerwise covering all utterences
bfre_ft_acc = [ 0.268, 0.320, 0.296, 0.32,     0.31, 0.34, 0.35,   0.305,  0.427,  0.349,      0.46, 0.54, 0.44, 0.48, 0.54] 
bfre_ft_acc = [i*100 for i in bfre_ft_acc]
aftr_ft_acc  = [0.29, 0.28, 0.44,  0.44,       0.44, 0.54, 0.62,      0.49,  0.55, 0.66,         0.76, 0.80, 0.80, 0.81,  0.84] 
aftr_ft_acc = [i*100 for i in aftr_ft_acc]
patterns = {'M04': '//', 'F03': '//', 'M12': '//', 'M01': '//', 
            'M07': '-', 'F02': '-', 'M16': '-', 
            'M05': '\\\\', 'M11': '\\\\', 'F04': '\\\\', 
            'M09': 'X', 'M14': 'X', 'M08': 'x', 'M10': 'X', 'F05': 'X'}

hatches = [patterns[spkr] for spkr in spkrs]

plt.bar(spkrs, aftr_ft_acc, color='#EA5739', hatch=hatches, label= 'After ft')
plt.bar(spkrs, bfre_ft_acc, color='#4BB05C', hatch=hatches, label= 'before ft')

a = mpatches.Patch(color='#EA5739', label= 'After ft')
b = mpatches.Patch(color='#4BB05C', label= 'before ft')
c = mpatches.Patch( facecolor='#FEFFBE', hatch='///' ,edgecolor='black',label= 'very low')
d = mpatches.Patch( facecolor='#FEFFBE', hatch='---' ,edgecolor='black',label= 'low')
e = mpatches.Patch( facecolor='#FEFFBE', hatch='\\\\' ,edgecolor='black',label= 'mid')
f = mpatches.Patch( facecolor='#FEFFBE', hatch='XX' ,edgecolor='black',label= 'high')
plt.legend(handles=[a,b,c, d, e, f],  loc=2)

plt.title('Accuracy - before & after Finetuning for dysarthric speakers')
plt.xlabel('Dysarthric speakers (based on increasing order of intelligibility)')
plt.ylabel('Accuracy (%)')
plt.show()