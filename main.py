from math import exp

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from json import loads

app = Flask(__name__)

cars_model = pickle.load(open('./models/cars/model.pkl', 'rb'))
cars_pipe = pickle.load(open('./models/cars/pipe.pkl', 'rb'))
[cars_categories, cars_numbers] = pickle.load(open('./models/cars/attrs.pkl', 'rb'))

vk_df = pd.read_csv('./models/vk/dataset.csv')
vk_categories = [
    {'name': 'Группа', 'values': sorted(vk_df['Группа'].unique())},
    {'name': 'Пол', 'values': ['Любой', *sorted(vk_df['Пол'].unique())]},
    {'name': 'Возраст', 'values': ['Любой', '18+']}
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/cars', methods=['GET'])
def cars():
    return render_template('cars.html', categories=cars_categories, numbers=cars_numbers)


@app.route('/cars/predict', methods=['POST'])
def cars_predict():
    df = pd.DataFrame(request.json, index=[0])
    pred = cars_model.predict(cars_pipe.transform(df))
    return jsonify(exp(pred[0]))


@app.route('/vk', methods=['GET'])
def vk():
    return render_template('vk.html', categories=vk_categories)


@app.route('/vk/predict', methods=['POST'])
def vk_predict():
    group = request.json['Группа']
    gender = request.json['Пол']
    age = request.json['Возраст']

    result = vk_df[vk_df['Группа'] == group]

    if gender != 'Любой':
        result = result[result['Пол'] == gender]

    if age == '18+':
        result = result[(result['Возраст'] > 18) | (result['Возраст'] == 0)]

    return jsonify(loads(result.to_json(orient='records')))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
