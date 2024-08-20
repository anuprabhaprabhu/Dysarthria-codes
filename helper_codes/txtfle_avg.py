
import sys,os

# Define the path to your file
fldr_path = sys.argv[1]  # Replace 'your_file.txt' with the path to your actual file

files = [i for i in os.listdir(fldr_path) if i.endswith('.txt')]

for fle in files:
    file_path = os.path.join(fldr_path,fle)
    acc_values = []

    # Read the file and extract acc values
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(", ")
            # print(parts)
            for part in parts:
                if part.startswith("acc="):
                    # Extract the acc value and convert it to float
                    acc_value_str = part.split("=")[1].split(",")[0]
                    acc_value = float(acc_value_str)
                    acc_values.append(acc_value)

    # Check if there are any acc values to avoid division by zero
    if acc_values:
        # Calculate the average of the acc values
        average_acc = sum(acc_values) / len(acc_values)
        print(f"Average acc for {fle}: {average_acc}")
    else:
        print("No acc values found in the file.")


print('Completted !!!!!')