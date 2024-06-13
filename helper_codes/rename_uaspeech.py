import os, sys

# folder_path = sys.argv[1]
input_file = sys.argv[1]
train_file = os.path.join(os.path.dirname(input_file), 'train.txt')
test_file = os.path.join(os.path.dirname(input_file), 'test.txt')

def convert_paths(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    converted_lines = []

    for line in lines:
        path, label = line.strip().split('\t')
        filename = os.path.basename(path)
        folder_name = filename.split('_')[0]  
        new_path = os.path.join('UASpeech','UASpeech_noisereduce','dysar', folder_name, filename)
        converted_line = f"{new_path}\t{label}\t0"   # 1-ctrl, 0-dysar
        converted_lines.append(converted_line)

    with open(output_file, 'w') as file:
        for line in converted_lines:
            file.write(line + '\n')

def train_dev_test(input_file, train_file, test_file):
    train_set = []
    test_set = []
    with open(input_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        path, label, _= line.strip().split('\t')
        filename = os.path.basename(path)
        folder_name = filename.split('_')[1]  
        # new_path = os.path.join('UASpeech','UASpeech_noisereduce','dysar', folder_name, filename)
        # converted_line = f"{new_path}\t{label}\t0"   # 1-ctrl, 0-dysar
        if folder_name=='B2':  #text set
            # print(folder_name, 'test')
            test_set.append(line)
        else:
            # print(folder_name)
            train_set.append(line)

    with open(train_file, 'a') as f1, open(test_file, 'a') as f2:
        for line in train_set:
            f1.write(line )
        for line in test_set:
            f2.write(line )


train_dev_test(input_file, train_file, test_file)
print(f"Processed {input_file} ") #to {output_file}")

# for filename in os.listdir(folder_path):
#     print(filename)
#     if filename.endswith('.txt'):
#         input_file = os.path.join(folder_path, filename)
#         output_file = os.path.join(folder_path, 'trans_' + filename)
#         train_file = os.path.join(folder_path, 'train_' + filename)
#         test_file = os.path.join(folder_path, 'test_' + filename)
#         # convert_paths(input_file, output_file)
#         train_dev_test(input_file, train_file, test_file)
#         print(f"Processed {input_file} ") #to {output_file}")

print('Completed !!!!!')
