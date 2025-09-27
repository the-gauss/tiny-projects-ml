from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    species = data['Species']
    length1 = float(data['Length1'])
    length2 = float(data['Length2'])
    length3 = float(data['Length3'])
    height = float(data['Height'])
    width = float(data['Width'])

    # Creating the input DataFrame
    input_data = pd.DataFrame({
        'Species': [species],
        'Length1': [length1],
        'Length2': [length2],
        'Length3': [length3],
        'Height': [height],
        'Width': [width]
    })
    
    prediction = model.predict(input_data)

    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
