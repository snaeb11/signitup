# train_model.py

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

EXPECTED_LENGTH = 42
data = []
labels = []

for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == EXPECTED_LENGTH:
        data.append(d)
        labels.append(l)

data = np.array(data)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = tf.keras.utils.to_categorical(labels_encoded)

# Save label encoder for Android decoding
joblib.dump(le, 'label_encoder.pkl')

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, stratify=labels_encoded, random_state=42)

# Define Keras model
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(labels_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, validation_split=0.1, epochs=30, batch_size=16)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("asl_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("asl_model.tflite", "wb") as f:
    f.write(tflite_model)
