import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/train.csv', encoding='ISO-8859-1')

# Handle missing values
df['text'] = df['text'].fillna('')  # or df = df.dropna(subset=['text'])

X = df['text']
y = df['sentiment']

vectorizer = CountVectorizer(max_features=2000)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

print(predict_sentiment("I am varun "))
