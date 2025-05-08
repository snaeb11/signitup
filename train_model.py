import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Clean inconsistent samples
EXPECTED_LENGTH = 42  # 21 landmarks * 2 (x and y)
cleaned_data = []
cleaned_labels = []

for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == EXPECTED_LENGTH:
        cleaned_data.append(d)
        cleaned_labels.append(l)
    else:
        print(f"Skipping sample with length {len(d)}")

data = np.asarray(cleaned_data)
labels = np.asarray(cleaned_labels)

# Optional: check label distribution
unique, counts = np.unique(labels, return_counts=True)
print("Label distribution:", dict(zip(unique, counts)))

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('\n{}% of samples were classified correctly!'.format(score * 100))
print("\nDetailed classification report:\n")
print(classification_report(y_test, y_predict))

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)