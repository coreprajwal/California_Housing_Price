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

if __name__=="__main__":
       app.run(debug=True)