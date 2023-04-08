import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
import data_gen

frames_dir = "MALL/frames/frames"
labels_file = "MALL/labels.csv"

# Load the labels file
labels_df = pd.read_csv(labels_file)

# Load the trained model
model = tf.keras.models.load_model("cnn_people_counter.h5")

# Load and preprocess the test image
test_image_path = os.path.join(frames_dir, "seq_000720.jpg")
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (256, 256)) / 255.0

# Get the id from the test image file name
id = int(os.path.splitext(os.path.basename(test_image_path))[0].split("_")[1])

# Find the corresponding count from the labels dataframe
count = labels_df.loc[labels_df['id'] == id]['count'].values[0]

# Predict the count on the test image
test_image = np.expand_dims(test_image, axis=0)
predicted_count = model.predict(test_image)[0][0]

# Display the test image with the predicted count and the actual count
plt.imshow(test_image[0])
plt.title(f"Predicted count: {predicted_count:.2f}, Actual count: {count}")
plt.show()

print(f"Predicted count: {predicted_count}, Actual count: {count}")

# Evaluate the model on a test set
# test_dataset = data_gen.data_generator(batch_size=8)
# test_steps = len(labels_df) // 8
# test_loss, test_mae = model.evaluate(test_dataset, steps=test_steps)

# Compute the predictions on the test set
# y_true = np.concatenate([next(test_dataset)[1] for i in range(test_steps)])
# y_pred = model.predict(test_dataset, steps=test_steps).flatten()

# Compute the metrics
# mae = mean_absolute_error(y_true, y_pred)
# mse = mean_squared_error(y_true, y_pred)
# relative_error = mae / np.mean(y_true) * 100
#
# print(f"Test loss: {test_loss:.4f}")
# print(f"Test MAE: {test_mae:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"MSE: {mse:.4f}")
# print(f"Relative Error: {relative_error:.2f}%")
