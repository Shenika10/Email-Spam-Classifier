from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required nltk packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# Load saved files
model = pickle.load(open('model/spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

ps = PorterStemmer()


def transform_text(text):

    text = text.lower()
    text = word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = ""

    if request.method == 'POST':

        message = request.form['message']

        transformed_message = transform_text(message)

        vector_input = vectorizer.transform(
            [transformed_message]
        )

        result = model.predict(vector_input)[0]

        if result == 1:
            prediction = "SPAM ❌"
        else:
            prediction = "HAM ✅"

    return render_template(
        'index.html',
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)