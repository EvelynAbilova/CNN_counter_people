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

# Load the trained model
model = tf.keras.models.load_model("cnn_people_counter.h5")

# Load and preprocess the test image
test_image_path = os.path.join(frames_dir, "seq_000905.jpg")
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (256, 256)) / 255.0

# Predict the count on the test image
test_image = np.expand_dims(test_image, axis=0)
predicted_count = model.predict(test_image)[0][0]

# Display the test image with the predicted count
plt.imshow(test_image[0])
plt.title(f"Predicted count: {predicted_count:.2f}")
plt.show()

print(f"Predicred count: {predicted_count}")
