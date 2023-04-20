import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import data_gen

MALL_DIR = "MALL/"
frames_dir = "MALL/frames/frames"
labels_file = "MALL/labels.csv"

# Load the labels file
labels_df = pd.read_csv(labels_file)

# Load the trained model
model = tf.keras.models.load_model("cnn_people_counter.h5")

# Load and preprocess the test image
test_image_path = os.path.join(frames_dir, "seq_000001.jpg")
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
test_dataset = data_gen.data_generator(batch_size=8)
test_steps = len(labels_df) // 8
test_loss, test_mae = model.evaluate(test_dataset, steps=test_steps)

# Compute the predictions on the test set
y_true = np.concatenate([next(test_dataset)[1] for i in range(test_steps)])
y_pred = model.predict(test_dataset, steps=test_steps).flatten()
sorted_indices = np.argsort(y_pred)

# Compute the metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
relative_error = mae / np.mean(y_true) * 100

print(f"Test loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Relative Error: {relative_error:.2f}%")

# Load the test dataset
test_dataset = data_gen.data_generator(batch_size=8)

# Predict on the test dataset
y_true = []
y_pred = []

for i, (X, y) in enumerate(test_dataset):
    y_true += y.tolist()
    y_pred += model.predict(X).flatten().tolist()

    if (i + 1) * 8 >= len(labels_df):
        break

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute the absolute errors
errors = np.abs(y_pred - y_true)

# Save the predicted and actual counts for each test sample
results = []
for i in range(len(y_true)):
    img_path = os.path.join(MALL_DIR, 'frames', 'frames', 'seq_{:06d}.jpg'.format(i+1))
    actual_count = int(y_true[i])
    predicted_count = int(round(y_pred[i]))
    results.append((img_path, predicted_count, actual_count))

# Sort the results by the difference between actual count and predicted count
results.sort(key=lambda x: abs(x[2] - x[1]))

# Select the best and worst results based on prediction accuracy
best_result = results[0]
worst_result = results[-1]

# Calculate the prediction accuracy for the best and worst results
best_accuracy = (1 - abs(best_result[1] - best_result[2]) / best_result[1]) * 100
worst_accuracy = (1 - abs(worst_result[2] - worst_result[1]) / worst_result[2]) * 100

# Display the best and worst results
plt.figure(figsize=(10, 5))

# Display the best result
plt.subplot(1, 2, 1)
img = cv2.imread(best_result[0], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title('Best score\nActual count: {}\nPredicted count: {}\nAccuracy: {:.2f}%'.format(
    best_result[2], best_result[1], best_accuracy))

# Display the worst result
plt.subplot(1, 2, 2)
img = cv2.imread(worst_result[0], cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.title('Worst score\nActual count: {}\nPredicted count: {}\nAccuracy: {:.2f}%'.format(
    worst_result[2], worst_result[1], worst_accuracy))

plt.show()
