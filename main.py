import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
 
# Load dataset
digits = load_digits()
X, y = digits.data, digits.target
print("Dataset shape:", X.shape, "Classes:", np.unique(y))
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
 
# Build ANN (Multi-layer Perceptron)
clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                     max_iter=300, random_state=42, verbose=False)
clf.fit(X_train_s, y_train)
 
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))
 
# Save training loss curve
plt.figure(figsize=(6,4))
plt.plot(clf.loss_curve_, color="#1B2A4A", linewidth=2)
plt.title("Training Loss Curve — Handwritten Digit Recognition (ANN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.close()
 
# Sample digits grid
fig, axes = plt.subplots(2, 8, figsize=(11, 3.2))
rng = np.random.RandomState(7)
idxs = rng.choice(len(X), 16, replace=False)
for ax, idx in zip(axes.ravel(), idxs):
    ax.imshow(digits.images[idx], cmap="gray")
    ax.set_title(str(digits.target[idx]), fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Images from the Handwritten Digits Dataset")
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=150)
plt.close()
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Test Set Predictions")
plt.colorbar()
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
 
# Prediction samples grid (correct vs incorrect)
fig, axes = plt.subplots(2, 8, figsize=(11, 3.2))
test_idxs = rng.choice(len(X_test), 16, replace=False)
for ax, idx in zip(axes.ravel(), test_idxs):
    img = X_test[idx].reshape(8, 8)
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    ax.imshow(img, cmap="gray")
    color = "green" if true_label == pred_label else "red"
    ax.set_title(f"P:{pred_label} / T:{true_label}", fontsize=9, color=color)
    ax.axis("off")
plt.suptitle("Model Predictions on Test Samples (P=Predicted, T=True)")
plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.close()
 
print("All artifacts saved.")
