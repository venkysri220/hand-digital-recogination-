# Load dataset
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset Shape:", X.shape)
print("Classes:", np.unique(y))

# Show sample images
fig, axes = plt.subplots(2, 8, figsize=(10, 4))

random = np.random.RandomState(7)
indices = random.choice(len(X), 16, replace=False)

for ax, idx in zip(axes.ravel(), indices):
    ax.imshow(digits.images[idx], cmap="gray")
    ax.set_title(digits.target[idx])
    ax.axis("off")

plt.suptitle("Sample Images from the Handwritten Digits Dataset")
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100)

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Loss Curve
plt.figure(figsize=(6,4))
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# Prediction Samples
fig, axes = plt.subplots(2, 8, figsize=(10,4))

sample = random.choice(len(X_test), 16, replace=False)

for ax, idx in zip(axes.ravel(), sample):
    ax.imshow(X_test[idx].reshape(8,8), cmap="gray")
    ax.set_title(f"P:{y_pred[idx]} T:{y_test[idx]}")
    ax.axis("off")

plt.suptitle("Predictions")
plt.show()