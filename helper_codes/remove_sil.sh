
# sox in.wav out.wav silence 1 .3 1% reverse silence 1 .3 1% reverse



#!/bin/bash

# Define the input and output directories
input_dir="$1"
output_dir="$2"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Find all .wav files in the input directory and process each one
find "$input_dir" -type f -name '*.wav' | while read -r file; do
    # Define the output file path
    output_file="$output_dir/${file#$input_dir/}"
    output_dir_path=$(dirname "$output_file")

    # Ensure the output file's directory exists
    mkdir -p "$output_dir_path"

    # Apply the sox command
    sox "$file" "$output_file" silence 1 .3 1% reverse silence 1 .3 1% reverse
done
