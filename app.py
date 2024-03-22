from flask import Flask, render_template, request
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = str(text)

    # Remove 'READ MORE' if found
    text = text.replace("READ MORE", "")

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r":\)|:\(|:\D|:\S", "", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_text = [word for word in words if word not in stop_words]
    filtered_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return " ".join(filtered_text)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prediction", methods=["get", "post"])
def prediction():
    text = [request.form.get("input_text")]
    text1=request.form.get("input_text")
    text_clean = [preprocess_text(text) for text in text]

    model = joblib.load("models/naive_bayes.pkl")
    prediction = model.predict(text_clean)
    return render_template("output.html", prediction=prediction[0] ,text1=text1)


if __name__ == "__main__":
    app.run(port=5000, debug=True)