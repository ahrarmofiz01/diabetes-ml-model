from flask import Flask, render_template, request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import uuid
import os

app = Flask(__name__)

# Load model
model = pickle.load(open('model/diabetes_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])
    prediction = model.predict(input_data)[0]

    bmi = float(request.form['BMI'])
    glucose = float(request.form['Glucose'])

    graph_name = f"{uuid.uuid4()}.png"
    graph_path = os.path.join("static", graph_name)

    # Create comparison bar charts
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["Your BMI", "Ideal BMI"], [bmi, 22], color=['orange', 'green'])
    ax[0].set_title("BMI Comparison")
    ax[0].set_ylim(0, max(bmi, 30))

    ax[1].bar(["Your Glucose", "Ideal Glucose"], [glucose, 90], color=['red', 'green'])
    ax[1].set_title("Glucose Level Comparison")
    ax[1].set_ylim(0, max(glucose, 200))

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    return render_template("result.html", prediction=prediction, graph=graph_name)

if __name__ == '__main__':
    app.run(debug=True)
