from flask import Flask, render_template, request, redirect

import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
import re
import pickle

app = Flask(__name__)

nltk.download('stopwords')
stop = stopwords.words('spanish')


@app.route('/')
def hello_world():
    return render_template('first_app.html')


if __name__ == '__main__':
    app.run()


@app.route('/feelings', methods=['POST', 'GET'])
def login():
    if request.method == 'GET':
        return render_template('first_app.html', title='Sentimientos')
    else:
        feeling: str = request.form.get("feeling")
        classification = get_feeling_type(feeling)
        return f"<h1>El sentimiento '{feeling}'' es {classification} </h1>"


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def get_feeling_type(feeling):
    clf = pickle.load(open(os.path.join('data/twitter_classifier/pkl_objects', 'classifier.pkl'), 'rb'))

    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2 ** 21,
                             preprocessor=None,
                             tokenizer=tokenizer)

    label = {-1: 'Sin sentimiento', 0: 'Neutro', 1: 'Positivo', 2: 'Negativo'}

    text_convert = vect.transform([feeling])
    return label[clf.predict(text_convert)[0]]
