import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os

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

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
batch_size = 32
train_steps = len(labels_df) // batch_size
EPOCHS = 10

train_generator = data_generator(batch_size=batch_size)

model.fit(train_generator, steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1)

# Evaluate the model on a test set
test_dataset = data_generator(batch_size=32)
test_steps = len(labels_df) // 32
test_loss, test_mae = model.evaluate(test_dataset, steps=test_steps)

print(f"Test loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

model.save('cnn_people_counter.h5')

