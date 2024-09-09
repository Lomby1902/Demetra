from flask import Flask, request, jsonify
import mlflow
import numpy as np
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_25 = 'models:/Ridge_CD_25/1'
model_50 = 'models:/Ridge_CD_50/1'
model_75 = 'models:/Ridge_CD_75/1'


# Predict on a Pandas DataFrame.
app = Flask(__name__)


@app.route('/response', methods=['GET'])
def response():
    data = request.get_json()
    model = data["model"]
    values = np.array(data["values"])
    loaded_model= mlflow.pyfunc.load_model(f"models:/{model}/1")
    result = loaded_model.predict(values)
    return jsonify({"message":result.tolist()})

if __name__ == '__main__':
    print("Starting server")
    app.run(debug=True, port=5001)
