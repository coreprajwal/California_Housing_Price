from flask import request,Flask,render_template,jsonify,render_template
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)

## Load the model
regmodel=pickle.load(open("project1.pkl","rb"))

@app.route("/")
def home():
       return render_template("home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
       scaling=pickle.load(open("scalerpickle.pkl","rb"))
       rawdata=request.json["data"]
       data=np.array([list(rawdata.values())]).reshape(1,-1)
       scaleddata=scaling.transform(data)
       prediction=regmodel.predict(scaleddata)
       return jsonify(prediction.tolist()[0][0])

@app.route("/predict",methods=["POST"])
def predict():
       formdata=np.array([float(x) for x in request.form.values()]).reshape(1,-1)
       print("data is:-",formdata,"total len is",len(formdata))
       scaling=pickle.load(open("scalerpickle.pkl","rb"))
       scaledform=scaling.transform(formdata)
       prediction=regmodel.predict(scaledform)
       return render_template("home.html",predictionform=prediction[0])


if __name__=="__main__":
       app.run(debug=True)