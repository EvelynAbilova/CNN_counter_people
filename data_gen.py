import cv2
import os
import pandas as pd
import numpy as np

# Load the dataset
frames_dir = "MALL/frames/frames"
labels_file = "MALL/labels.csv"

# Load the labels file
labels_df = pd.read_csv(labels_file)

# Define a data generator to load and preprocess images
def data_generator(batch_size=8):
    while True:
        # Shuffle the dataset
        shuffled_df = labels_df.sample(frac=1)

        # Loop over the shuffled dataset in batches
        for i in range(0, len(shuffled_df), batch_size):
            batch_df = shuffled_df.iloc[i:i+batch_size]
            batch_images = []
            batch_counts = []
            for _, row in batch_df.iterrows():
                # Load and preprocess the image
                image_path = os.path.join(frames_dir, f"seq_{row['id']:06d}.jpg")
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256)) / 255.0

                # Add the image and count to the batch
                batch_images.append(image)
                batch_counts.append(row['count'])

            yield np.array(batch_images), np.array(batch_counts)
