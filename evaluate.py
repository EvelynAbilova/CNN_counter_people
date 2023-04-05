import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt

# Load the dataset
frames_dir = "MALL/frames/frames"
labels_file = "MALL/labels.csv"

# Load the labels file
labels_df = pd.read_csv(labels_file)

# Define a data generator to load and preprocess images
def data_generator(batch_size=32):
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

# Load the trained model
model = tf.keras.models.load_model("cnn_people_counter.h5")

# Load and preprocess the test image
test_image_path = os.path.join(frames_dir, "seq_000099.jpg")
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (256, 256)) / 255.0

# Predict the count on the test image
test_image = np.expand_dims(test_image, axis=0)
predicted_count = model.predict(test_image)[0][0]

# Display the test image with the predicted count
plt.imshow(test_image[0])
plt.title(f"Predicted count: {predicted_count:.2f}")
plt.axis("off")
plt.show()

print(f"Predicred count: {predicted_count}")