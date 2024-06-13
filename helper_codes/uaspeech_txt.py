import os, sys


folder_path = sys.argv[1]
def convert_content(input_file, output_file):
    # Read the content of the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Initialize a list to hold the converted lines
    converted_lines = []

    # Process the lines
    for i in range(1, len(lines), 3):
        # Extract the filename and label from the lines
        filename = lines[i].strip().strip('"')
        label = lines[i+1].strip().lower()

        # Create the formatted string and append to the list
        converted_line = f"{filename.strip('*/').replace('.lab','.wav')}\t{label}"
        converted_lines.append(converted_line)

    # Write the converted lines to the output file
    with open(output_file, 'w') as file:
        for line in converted_lines:
            file.write(line + '\n')


# Get a list of all text files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.mlf'):
        input_file = os.path.join(folder_path, filename)
        output_file = os.path.join(folder_path, filename.strip('.mlf')+'.txt')
        convert_content(input_file, output_file)
        print(f"Processed {filename}")

print('Completed !!!!!')
