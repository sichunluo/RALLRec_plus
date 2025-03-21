#!/bin/bash

# Bash script to download MovieLens 1M dataset

# Set the URL for the dataset (MovieLens 1M dataset)
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# Set the output filename
OUTPUT_FILE="ml-1m.zip"

# Directory to store the dataset
DATASET_DIR="../data/ml-1m/raw_data"

# Check if wget or curl is installed
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null
then
    echo "Please install wget or curl to download the dataset."
    exit 1
fi

# Create a directory for the dataset if it doesn't exist
if [ ! -d "$DATASET_DIR" ]; then
    mkdir -p "$DATASET_DIR"
    mkdir -p "$DATASET_DIR"
fi

# Download the dataset using wget or curl
if command -v wget &> /dev/null
then
    echo "Downloading MovieLens 1M dataset using wget..."
    wget -O "$DATASET_DIR/$OUTPUT_FILE" "$URL"
elif command -v curl &> /dev/null
then
    echo "Downloading MovieLens 1M dataset using curl..."
    curl -o "$DATASET_DIR/$OUTPUT_FILE" "$URL"
fi

# Unzip the dataset
echo "Unzipping the dataset..."
unzip "$DATASET_DIR/$OUTPUT_FILE" -d "$DATASET_DIR"

# Remove the zip file after extraction
echo "Cleaning up..."
rm "$DATASET_DIR/$OUTPUT_FILE"

echo "Download and extraction complete!"
