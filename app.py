# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['TSH'])
        glucose = int(request.form['T3'])
        bp = int(request.form['TT4'])
        st = int(request.form['T4U'])
        insulin = int(request.form['FTI'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin,age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)