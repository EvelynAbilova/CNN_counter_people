import pandas as pd
import tensorflow as tf
import data_gen

# Load the dataset
frames_dir = "MALL/frames/frames"
labels_file = "MALL/labels.csv"

# Load the labels file
labels_df = pd.read_csv(labels_file)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
batch_size = 32
train_steps = len(labels_df) // batch_size
EPOCHS = 10

train_generator = data_gen.data_generator(batch_size=batch_size)

model.fit(train_generator, steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1)

# Evaluate the model on a test set
test_dataset = data_gen.data_generator(batch_size=32)
test_steps = len(labels_df) // 32
test_loss, test_mae = model.evaluate(test_dataset, steps=test_steps)

print(f"Test loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

model.save('cnn_people_counter.h5')

