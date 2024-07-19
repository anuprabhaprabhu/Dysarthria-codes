import sys, os
from collections import Counter
from collections import defaultdict
import pandas as pd

### Use this code -- for features 
src = sys.argv[1]
dest = sys.argv[2]
df = pd.read_csv(src)
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
# print(uni_spkrs)


for i in uni_spkrs:
    print(f'Processing for spkr {i}')
    spkr_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == i]
    ### when processing KWS process files
    # spkr_df = spkr_df[spkr_df['label']==1]
    spkr_df.to_csv(os.path.join(os.path.dirname(dest),f'{i}.csv'), sep=',', index=False)
    # sys.exit()
print('completed !!!!!')