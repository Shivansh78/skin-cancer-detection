# import pickle
# import numpy as np
# import tensorflow as tf
# import xgboost as xgb
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Load the first Deep Learning Model (h5 file)
# dl_model_1 = tf.keras.models.load_model("./efficientnet_b0.h5", compile=False)  # Update with actual file name

# # Load the second Deep Learning Model (h5 file)
# dl_model_2 = tf.keras.models.load_model("./efficientnet_b3.h5", compile=False)  # Update with actual file name

# # Load the XGBoost Model (pkl file)
# with open("./xgb_ensemble.pkl", "rb") as file:
#     xgb_model = pickle.load(file)

# @app.route("/")
# def home():
#     return "Ensemble Learning Model API is Running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()  # Receive input JSON
#         features = np.array(data["features"]).reshape(1, -1)  # Reshape input

#         # Predictions using both Deep Learning Models
#         dl_prediction_1 = dl_model_1.predict(features)
#         dl_prediction_2 = dl_model_2.predict(features)

#         # Prediction using XGBoost Model
#         dmatrix = xgb.DMatrix(features)  # Convert to DMatrix for XGBoost
#         xgb_prediction = xgb_model.predict(dmatrix)

#         # Ensemble Method (Averaging predictions from all three models)
#         final_prediction = (dl_prediction_1.flatten() + dl_prediction_2.flatten() + xgb_prediction) / 3

#         return jsonify({"deep_learning_prediction_1": dl_prediction_1.tolist(),
#                         "deep_learning_prediction_2": dl_prediction_2.tolist(),
#                         "xgboost_prediction": xgb_prediction.tolist(),
#                         "final_prediction": final_prediction.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)
import pickle
import numpy as np
import tensorflow as tf
import xgboost as xgb
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load Models
dl_model_1 = tf.keras.models.load_model("./efficientnet_b0.h5", compile=False)
dl_model_2 = tf.keras.models.load_model("./efficientnet_b3.h5", compile=False)

with open("./xgb_ensemble.pkl", "rb") as file:
    xgb_model = pickle.load(file)

@app.route("/")
def home():
    return "Ensemble Learning Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Read the image file and preprocess it
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predictions using both Deep Learning Models
        dl_prediction_1 = dl_model_1.predict(image_array)
        dl_prediction_2 = dl_model_2.predict(image_array)

        # Flatten the features for XGBoost
        features = image_array.flatten().reshape(1, -1)
        dmatrix = xgb.DMatrix(features)
        xgb_prediction = xgb_model.predict(dmatrix)

        # Ensemble Method (Averaging predictions)
        final_prediction = (dl_prediction_1.flatten() + dl_prediction_2.flatten() + xgb_prediction) / 3

        return jsonify({
            "deep_learning_prediction_1": dl_prediction_1.tolist(),
            "deep_learning_prediction_2": dl_prediction_2.tolist(),
            "xgboost_prediction": xgb_prediction.tolist(),
            "final_prediction": final_prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

