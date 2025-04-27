from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dictionary mapping model names to their respective pickle file paths (relative)
model_files = {
    "Logistic Regression": os.path.join(current_dir, "Logistic_Regression.pkl"),
    "Decision Tree": os.path.join(current_dir, "Decision_Tree.pkl"),
    "Random Forest": os.path.join(current_dir, "Random_Forest.pkl"),
    "SVM": os.path.join(current_dir, "SVM.pkl"),
    "KNN": os.path.join(current_dir, "KNN.pkl")
}

# Load models
models = {}
for name, path in model_files.items():
    with open(path, 'rb') as file:
        models[name] = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logs = []
    predictions = {}
    chart_data = {}

    try:
        input_data = [float(request.form[key]) for key in request.form]
        logs.append("Input successfully received and converted.")
    except ValueError:
        logs.append("Invalid input! Please enter numbers.")
        return render_template('index.html', logs=logs)

    if (input_data[0] == 0 or input_data[3] == 0 or input_data[4] == 0):
        logs.append("Some inputs like age, cholesterol, or blood pressure cannot be zero.")
        return render_template('index.html', logs=logs)

    feature_names = ['age', 'sex', 'chest_pain_type', 'resting_bp_s', 'cholesterol',
                     'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
                     'exercise_angina', 'oldpeak', 'st_slope']

    input_df = pd.DataFrame([input_data], columns=feature_names)

    for name, model in models.items():
        prob = model.predict_proba(input_df)[0]
        prob_at_risk = prob[1]
        prob_no_risk = prob[0]
        label = "At Risk" if prob_at_risk > 0.5 else "No Risk"

        predictions[name] = {
            "label": label,
            "prob_at_risk": prob_at_risk,
            "prob_no_risk": prob_no_risk
        }

        chart_data[name] = prob_at_risk
        logs.append(f"{name} predicted: {label} with probabilities: At Risk={prob_at_risk:.2f}, No Risk={prob_no_risk:.2f}")

    return render_template('index.html',
                           predictions=predictions,
                           logs=logs,
                           chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)
