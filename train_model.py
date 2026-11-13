from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Remove unnecessary columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:\n")
print(df.info())

# Check for missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Check number of spam and ham messages
print("\nSpam/Ham Count:\n")
print(df['label'].value_counts())
# Convert labels to numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nAfter Encoding:\n")
print(df.head())

print("\nEncoded Label Count:\n")
print(df['label'].value_counts())
# Create stemmer object
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def transform_text(text):

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    text = word_tokenize(text)

    y = []

    # Keep only alphabets and numbers
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
print("\nText Cleaning Example:\n")
print(transform_text(df['message'][0]))
# Apply preprocessing to all messages
df['transformed_message'] = df['message'].apply(transform_text)

print("\nTransformed Dataset:\n")
print(df.head())
# Create TF-IDF object
tfidf = TfidfVectorizer(max_features=3000)

# Convert text into numerical vectors
X = tfidf.fit_transform(df['transformed_message']).toarray()

# Labels
y = df['label'].values

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)
# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
# Create model object
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("\nPrecision :", precision_score(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
# Save TF-IDF object
pickle.dump(tfidf, open('model/vectorizer.pkl', 'wb'))

# Save trained model
pickle.dump(model, open('model/spam_model.pkl', 'wb'))

print("\nModel Saved Successfully!")