from pathlib import Path
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

folder = Path(__file__).resolve().parent

# Load only the columns needed for sentiment prediction
df = pd.read_csv(folder / 'exampledataset2.csv', usecols=['text', 'sentiment'])
df = df.dropna().drop_duplicates()
X = df['text']
y = df['sentiment']

# Reserve 20% for testing, with both labels represented
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Learn vocabulary from training text only
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train and test the SVM
model = SVC()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)

# Simple comparison model: always choose the most common training label
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_vectorized, y_train)
baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test_vectorized))

print('=== NLP SENTIMENT ANALYSIS ===')
print(f'Training reviews: {len(X_train)}')
print(f'Testing reviews: {len(X_test)}')
print(f'SVM accuracy: {accuracy:.2%}')
print(f'Baseline accuracy: {baseline_accuracy:.2%}')
print('\nTest predictions:')
for text, actual, predicted in zip(X_test, y_test, predictions):
    print(f'{text} | Actual: {actual} | Predicted: {predicted}')

results = (
    f'Python NLP Sentiment Analysis\n'
    f'Records: {len(df)}\n'
    f'Training: {len(X_train)} | Testing: {len(X_test)}\n'
    f'SVM accuracy: {accuracy:.2%}\n'
    f'Baseline accuracy: {baseline_accuracy:.2%}\n'
    f'Seed: 42\n'
    f'Pandas: {pd.__version__}\nScikit-learn: {sklearn.__version__}\n'
    'Limitations: This dataset has only 20 reviews, so test accuracy is not reliable evidence of real-world performance.\n'
)
(folder / 'results.txt').write_text(results, encoding='utf-8')
print('\nResults saved to results.txt')
