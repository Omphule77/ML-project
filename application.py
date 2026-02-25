import pickle
import os
from flask import Flask,request,redirect,render_template,jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

appication=Flask(__name__)
app=appication

## import ridge and scalar model
ridge_model=pickle.load(open('Model/ridge.pkl','rb'))
scaler_model=pickle.load(open('Model/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    appication.run(host="0.0.0.0", port=port)