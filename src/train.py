from src.data import load_data
from src.feature import split_features_labels, build_vectorizer, fit_vectorizer,transform_text
from src.model import get_model
from src.evaluate import get_accuracy, get_confusion_matrix
from src.artifacts import save_artifacts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib 

#load data
df = load_data("synthetic_text_data.csv")
X, y = split_features_labels(df)

#split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#convert text data to numerical data
victorizer = build_vectorizer()
X_train_victorized = fit_vectorizer(victorizer, X_train)
X_test_victorized = transform_text(victorizer, X_test)

#build and train the model
model = get_model()
model.fit(X_train_victorized, y_train)
y_pred = model.predict(X_test_victorized)

#evaluation
accuracy = get_accuracy(y_test, y_pred)
conf_matrix = get_confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy *100}%')

class_labels = np.unique(y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Save artifacts
save_artifacts(model, victorizer)