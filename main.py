from math import exp

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('./models/cars/model.pkl', 'rb'))
pipe = pickle.load(open('./models/cars/pipe.pkl', 'rb'))
[categories, numbers] = pickle.load(open('./models/cars/attrs.pkl', 'rb'))


@app.route('/cars')
def cars():
    return render_template('cars.html', categories=categories, numbers=numbers)

@app.route('/cars/predict', methods=['POST'])
def cars_predict():
    df = pd.DataFrame(request.json, index=[0])
    pred = model.predict(pipe.transform(df))
    return jsonify(exp(pred[0]))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)