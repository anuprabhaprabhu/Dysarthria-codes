import sys, os
import pandas as pd 
from glob import glob

inp_dir = sys.argv[1]    # input fldr
txt_files_fldr = sys.argv[2]   # fldr to save .txt files

### generate transcript

for i in glob(inp_dir+'/*', recursive= True):
    for j in glob(i+'/*', recursive= True):
        b1 = os.path.basename(j)
        for k in glob(j+'/*', recursive= True):    
            # print(k+'/prompts/')
            # include a condition to check whetthe "wav_headMic" folder is also there --> taken care by checking the .wav and .txt pairs
            b2 = os.path.basename(k)
            transcript_path = os.path.join(txt_files_fldr, b1+'_'+b2+'_'+'transcript.txt')
            print(transcript_path)
            
            # Open the transcript file in append mode
            with open(transcript_path, 'a') as transcript_file:
                # Iterate over all .txt files in the source folder
                for txt_file in glob(os.path.join(k+'/prompts/', '*.txt')):
                    # print(txt_file)
                    
                    # check if the corresponding .wav file is present
                    wav_file = txt_file.replace('prompts','wav_headMic').replace('txt','wav')
                    # print(wav_file)
                    if os.path.isfile(wav_file):
                        # print('OKAY !!!')
                        # Read the content of the txt file
                        with open(txt_file, 'r') as file:
                            content = file.read()
                        transcript_file.write(f"{txt_file}\t{content}\n")
        

print("Completed!!")



        


