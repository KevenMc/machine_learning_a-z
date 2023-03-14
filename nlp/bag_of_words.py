"""Bag od words sentiment analysis"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf

# Name data source
DATA_SOURCE = 'Restaurant_Reviews.tsv'

# Load dataset from csv
dataset = pd.read_csv(DATA_SOURCE, delimiter="\t", quoting=3)
y = dataset.iloc[:, -1].values


#Clean text data
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
all_stopwords.remove("isn't")

def clean(text: str) -> list[str]:
    out_text = re.sub('[^a-zA-Z]', ' ', text)
    out_text = out_text.lower().split()
    out_text = [ps.stem(word) for word in out_text if not word in set(all_stopwords)]
    return ' '.join(out_text)

corpus = [clean(text) for text in dataset['Review']]

#Count vectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)


# Train GaussianNB model on data
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict test set
y_pred = classifier.predict(X_test)

# Confusion matrix and fscore
m = confusion_matrix(y_pred.reshape(len(y_pred), 1),
                     y_test.reshape(len(y_test), 1))
f = f1_score(y_pred.reshape(len(y_pred), 1),
             y_test.reshape(len(y_test), 1))
a = accuracy_score(y_pred.reshape(len(y_pred), 1),
                   y_test.reshape(len(y_test), 1))

print(f"Accuracy (fscore): {f}")
print(f"Accuracy (accuracy score): {a}")
print(f"Confusion matrix:\n {m}")


# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(1500,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='sigmoid'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with your input data
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
print("Tensorflow")

predicted_probabilities = model.predict(X_test)

threshold = 0.5
y_pred = np.where(predicted_probabilities > threshold, 1, 0)

# Confusion matrix and fscore
m = confusion_matrix(y_pred.reshape(len(y_pred), 1),
                     y_test.reshape(len(y_test), 1))
f = f1_score(y_pred.reshape(len(y_pred), 1),
             y_test.reshape(len(y_test), 1))
a = accuracy_score(y_pred.reshape(len(y_pred), 1),
                   y_test.reshape(len(y_test), 1))

print(f"Accuracy (fscore): {f}")
print(f"Accuracy (accuracy score): {a}")
print(f"Confusion matrix:\n {m}")
