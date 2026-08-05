# Handwritten Digit Recognition
Handwritten Digit Recognition using Python and Artificial Neural Networks

Project Overview

This project implements a Handwritten Digit Recognition System using Python and an Artificial Neural Network (ANN). The model is trained on the Scikit-learn Digits dataset, which contains 1,797 grayscale images of handwritten digits (0–9). The objective is to classify handwritten digit images with high accuracy using a Multi-Layer Perceptron (MLP) classifier.

This project was developed as part of the Python Programming for Artificial Intelligence internship.

---

Features

- Handwritten digit classification (0–9)
- Data preprocessing using "StandardScaler"
- Artificial Neural Network (MLPClassifier)
- Model evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Visualization of:
  - Training Loss Curve
  - Sample Digit Images
  - Model Predictions

---

Technologies Used

- Python 3.11
- NumPy
- Matplotlib
- Scikit-learn

---

Dataset

The project uses the Digits Dataset available in the Scikit-learn library.

- Total Samples: 1,797
- Image Size: 8 × 8 pixels
- Classes: 10 (Digits 0–9)

---

Project Workflow

1. Load the Digits dataset.
2. Split the dataset into training and testing sets.
3. Standardize the feature values using "StandardScaler".
4. Build an Artificial Neural Network using "MLPClassifier".
5. Train the model on the training dataset.
6. Predict handwritten digits from the test dataset.
7. Evaluate the model using accuracy, confusion matrix, and classification report.
8. Visualize the results.

---

Project Output

The project generates:

- Training Loss Curve
- Sample Digit Images
- Confusion Matrix
- Model Prediction Images
- Classification Report
- Accuracy Score

---

Results

The Artificial Neural Network achieved approximately 97% test accuracy on the Digits dataset, demonstrating effective handwritten digit recognition.

---

Future Improvements

- Convolutional Neural Networks (CNN)
- TensorFlow/Keras implementation
- Support for custom handwritten images
- Web application using Flask or Streamlit

---

References

- Scikit-learn Documentation
- NumPy Documentation
- Matplotlib Documentation
- Python Documentation

---

Author

Developed as part of the Python Programming for Artificial Intelligence Internship.
