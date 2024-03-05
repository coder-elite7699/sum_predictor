from flask import Flask, render_template, request
import joblib 
import pickle

app = Flask(__name__)


model = joblib.load('sum_model.pkl')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    num1 = request.form['num1']  
    num2 = request.form['num2']  

    if not num1 or not num2:
        return render_template('index.html', error_message='Please enter both numbers.')

    try:
        num1 = float(num1)
        num2 = float(num2)
    except ValueError:
        return render_template('index.html', error_message='Invalid input. Please enter valid numbers.')

    
    prediction = model.predict([[num1, num2]])

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
