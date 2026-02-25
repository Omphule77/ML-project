from flask import Flask,request,render_template
import pickle
import os

application = Flask(__name__)
app = application

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':

        ridge_model = pickle.load(open(os.path.join(BASE_DIR,'Model','ridge.pkl'),'rb'))
        scaler_model = pickle.load(open(os.path.join(BASE_DIR,'Model','scaler.pkl'),'rb'))

        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_data_scaled = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", port=port)