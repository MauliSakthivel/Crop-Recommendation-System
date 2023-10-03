from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained RandomForest model
model_path = 'RandomForest.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop_recommend():
    if request.method == 'POST':
        try:
            # Get input data from the form
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorous'])
            K = float(request.form['pottasium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Make a prediction using the loaded model
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(data)
            print("Prediction:", prediction)
            return render_template('crop-result.html', prediction=prediction[0])
            
        except Exception as e:
            print("Error:", e)
            return render_template('try_again.html')

    return render_template('crop.html')


if __name__ == '__main__':
    app.run(debug=True)
