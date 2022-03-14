from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
from sklearn import linear_model
import joblib

import re
from sklearn.feature_extraction.text import TfidfVectorizer


  

import flask
app = Flask(__name__)
######################################
#load model

clf = joblib.load('RF.pkl')
count_vect = joblib.load('tfidf.pkl')
    
######################
#load Arabic stop words

stops = open('stopwords2.txt',encoding = 'utf-8')
stop_words = stops.read().split('\n')


###################################################
#Clean Text
def clean_text(text, stop_words):
   
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'\...', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?«»:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[a-z]+', ' ', text)
    text = re.sub(r'[A-Z]+', ' ', text)
    text = text.split()
    text = [w for w in text if not w in stop_words]
    text = " ".join(text)

    return text

###################################################
#Clean emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


@app.route('/')
def index():
    return flask.render_template('index.html')

#Retrieve text and Prediction Country
@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    text = to_predict_list['text']
    
    text_clean1 = clean_text(text,stop_words)
    
    text_clean2=remove_emoji(text_clean1)
    pred = clf.predict(count_vect.transform([text_clean2]))
  
    if pred==0:
        prediction = "EG"
    elif pred ==1:
        prediction = "PL"
    elif pred ==2:
        prediction = "KW"
    elif pred ==3:
        prediction = "LY"
    elif pred ==4:
        prediction = "QA"
    elif pred ==5:
        prediction = "JO"
    elif pred ==6:
        prediction = "LB"
    elif pred==7:
        prediction = "SA"
    elif pred==8:
        prediction = "AE"
    elif pred==9:
        prediction = "SA"
    elif pred==10:
        prediction = "BH"
    elif pred==11:
        prediction = "OM"
    elif pred==12:
        prediction = "SY"
    elif pred==13:
        prediction = "DZ"
    elif pred==14:
        prediction = "IQ"
    elif pred==15:
        prediction = "SD"
    elif pred==16:
        prediction = "MA"
    elif pred==17:
        prediction = "YE"
    elif pred ==18:
        prediction = "TN" 
    else:
        prediction = "Model can not prediction" 


    return render_template('index.html', prediction='Country Prediction: {}'.format(prediction))


if __name__ == '__main__':

    app.run(debug=True)
    
