from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('savedmodel.sav', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html', result=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate input
        features = [request.form.get(key) for key in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        if None in features or '' in features:
            raise ValueError("All fields are required.")

        features = list(map(float, features))
        if len(features) != 4:
            raise ValueError("Invalid input format.")

        # Make prediction
        prediction = model.predict([features])[0] if model else "Model not available"
        return render_template('index.html', result=prediction, error=None)

    except ValueError as e:
        return render_template('index.html', result=None, error=str(e))
    except Exception as e:
        return render_template('index.html', result=None, error="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
