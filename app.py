import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Assuming your input data is in the format of a list of dictionaries
        # where each dictionary represents a row in the CSV
        predictions = []
        for row in data:
            # Preprocess the input data (You may need to adjust this based on your actual data)
            input_data = [
                int(row['AGE']), int(row['PackHistory']), 
                int(row['MWT1']), int(row['MWT2']), float(row['FEV1']), float(row['FVC']),
                int(row['CAT']), int(row['HAD']), float(row['SGRQ']),
                int(row['copd']), int(row['gender']), int(row['smoking'])
            ]
            # Make prediction using the loaded model
            prediction = model.predict([input_data])
            predictions.append(int(prediction[0]))
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
