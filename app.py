from flask import Flask, request, jsonify, render_template
import pickle
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

# Load model and vectorizer
tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
punct = set(string.punctuation)


def transform_text(text):
    text = text.lower()
    words = word_tokenize(text)
    filtered = []
    for word in words:
        if word.isalnum() and word not in stop_words and word not in punct:
            filtered.append(ps.stem(word))
    return " ".join(filtered)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    transformed = transform_text(message)
    vector_input = tfidf.transform([transformed]).toarray()
    result = model.predict(vector_input)[0]

    return jsonify({
        "result": "spam" if result == 1 else "ham",
        "is_spam": bool(result == 1)
    })


if __name__ == "__main__":
    app.run(debug=True)