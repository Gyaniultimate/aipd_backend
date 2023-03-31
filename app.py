import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
import tensorflow as tf
import keras, json
from fileinput import filename
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os.path import abspath



from keras.models import load_model

import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = load_model(abspath("./models/model_parkinsons_v0.h5"))
sc = StandardScaler()




def readSensorDataFromFile(f):
    data = pd.read_csv(f, sep='\t')
    data['classification'] = getClassForHealthyOrNot(f)
    sc.fit(data)
    return sc.transform(data.values)

def getClassForHealthyOrNot(filename):
    return 1

def produceImagesFromFile(file, image_height, offset=100):
    r = pd.DataFrame()

    # Width is 16 pixels
    d = readSensorDataFromFile(file)[:, 1:17]

    for i in range(0, d.shape[0], offset):
        if (i+image_height > d.shape[0]):
            continue
        r = pd.concat([r, pd.DataFrame(d[i:i+image_height])], axis=0)

    return r.values.reshape(-1, 16, image_height, 1), getClassForHealthyOrNot(file)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print("hello i'm here")

    f = request.files['file']
    print(f)

    d_x, label = produceImagesFromFile(file=f, image_height=240)
    prediction = model.predict(d_x)
    predictions=np.argmax(prediction,axis=1)


    predict_distribution = pd.Series(predictions).value_counts()
    predictClass = predict_distribution.idxmax()

    print("this is predict class : ")
    print(predictClass)
    finalPrediction = ""
    if(predictClass == 1):
        finalPrediction = True
    else:
        finalPrediction = False


    #return render_template("acknowledgement.html", name = f.filename)
    data = [
        {
    "modelId": 1,
    "modelName": "CNN",
    "modelAccuracy": 100,
    "modelResult": finalPrediction,
  },
  {
    "modelId": 2,
    "modelName": "CNN",
    "modelAccuracy": 100,
    "modelResult": finalPrediction,
  },
  {
    "modelId": 3,
    "modelName": "CNN",
    "modelAccuracy": 100,
    "modelResult": finalPrediction,
  },
  {
    "modelId": 4,
    "modelName": "CNN",
    "modelAccuracy": 100,
    "modelResult": finalPrediction,
  },
  {
    "modelId": 5,
    "modelName": "CNN",
    "modelAccuracy": 100,
    "modelResult": finalPrediction,
  },
  {
    "modelId": 6,
    "modelName": "CNN",
    "modelAccuracy": 100,
    "modelResult": finalPrediction,
  }
];
    
    return  jsonify({'data': data})

@app.route('/predict_api',methods=['POST'])
@cross_origin()
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.files['files']
    print(data.filename)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)