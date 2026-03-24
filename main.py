import pickle
import numpy as np
import pandas as pd

from flask import Flask,request,jsonify
from flask_cors import CORS
app=Flask(__name__)
CORS(app)

with open("Spam_model.pkl","rb") as model_file:
    model=pickle.load(model_file)
with open("vectorizer.pkl","rb") as vectorizer_file:
    vectorize=pickle.load(vectorizer_file)
@app.route("/")
def home():
    return "welcome to the spam email detection API is running."
@app.route("/predict",methods=["POST"])
def predict():
    try:
        #get the data in json format from the API request
        data=request.get_json()
        input_data=pd.DataFrame([data])
        if(not data):
            return jsonify({"error":"no input data is provided "}),400
        required_columns=["Body"]
        #check the missng columns in the input data
        if(not all(col in input_data.columns for col in required_columns)):
            return jsonify({"error":"missing required columns. required columns={required_columns}"}),400
        #vectorize the data
        vectorize_data=vectorize.transform(input_data["Body"])
        #make prediction
        prediction=model.predict(vectorize_data) 
        try:
            proba=model.predict_proba(vectorize_data)[0][1]
            confidence=float(proba*100) if prediction[0]==1 else float((1-proba)*100)
        except:
            confidence=95.0
        prediction=model.predict(vectorize_data)
        #response
        response={
            "prediction":"Spam" if prediction[0]==1 else "Ham",
            "is_spam":bool(prediction[0]==1),
            "confidence": confidence
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error":str(e)}),500
if __name__=="__main__":
    app.run(debug=True)
        