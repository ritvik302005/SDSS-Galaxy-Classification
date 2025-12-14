from flask import Flask, render_template, request
import pandas as pd
import joblib
import json

best_model = joblib.load('RandomForest2.pkl')
scaler = joblib.load('scaler.pkl')

with open('selected_features.json', 'r') as f:
    feature_names = json.load(f)

label_map = {0: "STARFORMING", 1: "STARBURST"}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    input_values = []
    for feature in feature_names:
        value = float(request.form[feature])
        input_values.append(value)

    input_df = pd.DataFrame([input_values], columns=feature_names)

    scaled_input = scaler.transform(input_df)

    pred = best_model.predict(scaled_input)[0]
    prediction_label = label_map.get(int(pred), str(pred))

    return render_template('inner-page.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
