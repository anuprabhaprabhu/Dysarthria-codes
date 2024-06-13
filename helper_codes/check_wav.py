import os, sys


input_file = sys.argv[1]
output_file = sys.argv[2]
path = sys.argv[3]

def check_wav_files(input_file, output_file, pth):

    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    files = []

    for line in lines:
        parts = line.strip().split('\t')
        wav_file_path = parts[0]
        if os.path.exists(os.path.join(pth, wav_file_path)):
            files.append(line)


    with open(output_file, 'w') as outfile:
        for file in files:
            outfile.write(f"{file}")



check_wav_files(input_file, output_file, path)
