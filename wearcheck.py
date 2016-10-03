from flask import Flask, jsonify, request, render_template
from flask_wtf import Form
from wtforms import fields, StringField
from wtforms.validators import Required, Length

import numpy as np
import pandas as pd
import json
import requests as r
import urllib
from keras.models import model_from_json
from keras.preprocessing import image
import re

#---------- MODEL IN MEMORY ----------------#

# CLASSIFIER MODEL
with open('model_files/incep_2_multi.json', 'r') as f:
    model_json = f.read()
model_json = model_json.replace('"gamma_regularizer": null, ', '').replace('"beta_regularizer": null, ', '')
model = model_from_json(model_json)
model.load_weights("model_files/incep_2_multi.h5")
print("Successful model load.")

# FILTER MODEL
with open('model_files/incep_filter.json', 'r') as f:
    model_json2 = f.read()
model_json2 = model_json2.replace('"gamma_regularizer": null, ', '').replace('"beta_regularizer": null, ', '')
model2 = model_from_json(model_json2)
model2.load_weights("model_files/incep_filter.h5")

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    return x

def filter_p(img_path, thresh = .5):
    x = preprocess_img(img_path)
    preds = model2.predict(x)
    if (1 - preds[0][0]) > .5:
        return True

def clothing_predict(img_path):
    categories = {0:'Bikini',
                    1:'Blouse',
                    2:'Button Down',
                    3:'Coat',
                    4:'Dress',
                    5:'Hoodie',
                    6:'Jacket',
                    7:'Plaid Outfit',
                    8:'Polo Shirt',
                    9: 'Shirt',
                    10: 'Shirtless',
                    11: 'Suit',
                    12: 'Sweater'}

    message = {0:"Lookin' good in that bikini!",
                    1: 'Cute blouse!',
                    2: "That's a sharp lookin' button down",
                    3: 'Sweet coat!',
                    4: 'Gorgeous dress!',
                    5: 'Dope hoodie!',
                    6: 'Great jacket!',
                    7: 'Way to rock the plaid!',
                    8: 'Nice polo!',
                    9: 'Cool shirt!',
                    10: 'Nice tan bro. Put a shirt on!',
                    11: "Slick suit!",
                    12: "That's a sweater."}
    x = preprocess_img(img_path)
    preds = model.predict(x)
    arg1 = preds.argsort()[0][::-1][0]
    arg2 = preds.argsort()[0][::-1][1]
    return (categories[arg1], preds[0][arg1], categories[arg2], preds[0][arg2], message[arg1])

def make_prediction(img_path, url):
    filt_check = filter_p(img_path)
    if filt_check:
        # Check with model for clothing type
        output = clothing_predict(img_path)
        results = { "score1": "{:.3f}".format(output[1]),
                    "score2": "{:.3f}".format(output[3]),
                    "outfit1": output[0],
                    "outfit2": output[2],
                    "msg": output[4],
                    "url": url}
        return results

#---------- Form Class -------------#

class PredictForm(Form):
    """Fields for Insta Handle/URL"""
    entry = StringField('Instagram Username or URL:')

    submit = fields.SubmitField('Submit')


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)
app.config.from_object("config")


@app.route("/", methods=('GET', 'POST'))
def index():
    """
    Web Page
    """

    form = PredictForm()
    entry = None
    im_count = 0

    deliver = {"Image1": None, "Image2": None, "Image3": None}

    if form.validate_on_submit():
        img_path = 'temp.jpg'
        # retrieve the submitted values
        entry = form.data['entry']
        url_check = re.search('http|www', entry)
        if url_check:
            try:
                urllib.request.urlretrieve(entry, img_path)
                results = make_prediction(img_path, entry)
                if results:
                    deliver['Image1'] = results
                    im_count += 1
                else:
                    im_count = -1
            except:
                im_count = -3
        else:
            # Get decision score for our example that came with the request
            insta = r.get('https://www.instagram.com/{}/media/'.format(entry))
            try:
                if insta.json()['items']:
                    urls = []
                    for pics in insta.json()['items']:
                        urls.append(pics['images']['standard_resolution']['url'])
                    for url in urls:
                        if im_count == 3:
                            break
                        urllib.request.urlretrieve(url, img_path)
                        results = make_prediction(img_path, url)
                        if results:
                            im_count += 1
                            deliver['Image{}'.format(im_count)] = results
                else:
                    im_count = -2
            except ValueError:
                im_count = -4


    # jsondeliver = jsonify(deliver)
    return render_template("index.html",
            form=form,
            variables=deliver,
            im_count=im_count)

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
