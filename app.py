from doctest import OutputChecker
from unicodedata import category
from flask import Flask, request, render_template
import pickle
import numpy as np
import config
import tweepy

#twiiter api authentication
auth = tweepy.OAuth1UserHandler(
   config.API_KEY, config.API_KEY_SECRET, config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET
)

api = tweepy.API(auth)


#Create Flask app
app = Flask(__name__)

#loading model files
model = pickle.load(open("trained_model.sav",'rb'))
tfidf = pickle.load(open("vectorizer.pickle",'rb'))


#Routes
@app.route('/')
def home():
    return render_template("index.html")
    # return "Hello"

@app.route('/prediction')
def prediction():
    return render_template("predict.html",category="",boiler_text="",text="")

@app.route('/predict', methods = ["POST"])
def predict():
    text = request.form['text']
    output = model.predict(tfidf.transform([text]).toarray())
    return render_template("predict.html",text=text,category=output[0],boiler_text="Category is: ")

@app.route('/generate_random')
def generate_random():
    public_tweets = api.home_timeline()
    return public_tweets

if __name__ == "__main__":
    app.run(debug=True)