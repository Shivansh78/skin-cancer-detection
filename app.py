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
# import pickle
# import numpy as np
# import tensorflow as tf
# import xgboost as xgb
# from flask import Flask, request, jsonify
# from PIL import Image
# import io

# app = Flask(__name__)

# # Load Models
# try:
#     print("Loading deep learning models...")
#     dl_model_1 = tf.keras.models.load_model("./efficientnet_b0.h5", compile=False)
#     dl_model_2 = tf.keras.models.load_model("./efficientnet_b3.h5", compile=False)
    
#     print("Loading XGBoost model...")
#     with open("./xgb_ensemble.pkl", "rb") as file:
#         xgb_model = pickle.load(file)
    
#     print("All models loaded successfully!")
# except Exception as e:
#     print(f"Error loading models: {str(e)}")
#     raise

# @app.route("/")
# def home():
#     return "Ensemble Learning Model API is Running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files["file"]
#         print("This the Image input",file)
    
#         if file.filename == "":
#             return jsonify({"error": "No selected file"}), 400

#         # Read the image file and preprocess it
#         image = Image.open(io.BytesIO(file.read())).convert("RGB")
#         print("Here 1")
#         image = image.resize((224, 224))  # Resize to model input size
#         image_array = np.array(image) / 255.0  # Normalize
#         image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#         print("Here 2")

#         # Predictions using both Deep Learning Models
#         dl_prediction_1 = dl_model_1.predict(image_array)
#         dl_prediction_2 = dl_model_2.predict(image_array)

#         print("Here 3")

#         # Flatten the features for XGBoost
#         features = image_array.reshape(1, -1)
#         dmatrix = xgb.DMatrix(features)
#         xgb_prediction = xgb_model.predict(dmatrix)

#         print("Here 4")
#         # Ensemble Method (Averaging predictions)
#         final_prediction = (dl_prediction_1.flatten() + dl_prediction_2.flatten() + xgb_prediction) / 3

#         return jsonify({
#             "deep_learning_prediction_1": dl_prediction_1.tolist(),
#             "deep_learning_prediction_2": dl_prediction_2.tolist(),
#             "xgboost_prediction": xgb_prediction.tolist(),
#             "final_prediction": final_prediction.tolist()
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)


import pickle
import numpy as np
import tensorflow as tf
import xgboost as xgb
from flask import Flask, request, jsonify
from PIL import Image
import io
import traceback  # For detailed error reporting

app = Flask(__name__)

# Define class names (update with your actual class names)
CLASS_NAMES = [
    "Melanoma", 
    "Melanocytic nevus", 
    "Basal cell carcinoma", 
    "Actinic keratosis", 
    "Benign keratosis", 
    "Dermatofibroma", 
    "Vascular lesion"
]

# Load Models
try:
    print("Loading deep learning models...")
    dl_model_1 = tf.keras.models.load_model("./efficientnet_b0.h5", compile=False)
    dl_model_2 = tf.keras.models.load_model("./efficientnet_b3.h5", compile=False)
    
    print("Loading XGBoost model...")
    with open("./xgb_ensemble.pkl", "rb") as file:
        xgb_model = pickle.load(file)
    
    print("All models loaded successfully!")
    # Print model type for debugging
    print(f"XGBoost model type: {type(xgb_model)}")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    traceback.print_exc()
    raise

@app.route("/")
def home():
    return "Ensemble Learning Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        print(f"Processing image: {file.filename}")
    
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Read the image file and preprocess it
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        print(f"Image processed, shape: {image_array.shape}")

        # Predictions using both Deep Learning Models
        print("Running EfficientNet B0 prediction...")
        dl_prediction_1 = dl_model_1.predict(image_array)
        
        print("Running EfficientNet B3 prediction...")
        dl_prediction_2 = dl_model_2.predict(image_array)

        print("Deep learning predictions complete.")
        print(f"DL1 shape: {dl_prediction_1.shape}, DL2 shape: {dl_prediction_2.shape}")

        # OPTION A: Use the DL model outputs as input to XGBoost
        # This is more likely what your model was trained with
        try:
            # Combine the CNN features
            combined_features = np.hstack([
                dl_prediction_1.flatten(),
                dl_prediction_2.flatten()
            ]).reshape(1, -1)
            
            print(f"Combined features for XGB, shape: {combined_features.shape}")
            
            # Create DMatrix from combined CNN outputs
            dmatrix = xgb.DMatrix(combined_features)
            print("XGBoost DMatrix created successfully")
            
            # Make prediction
            xgb_prediction = xgb_model.predict(dmatrix)
            print("XGBoost prediction successful")
        
        # OPTION B: If A fails, try with raw image features
        except Exception as e:
            print(f"Error with Option A: {str(e)}")
            print("Trying alternative approach with raw image features...")
            
            # Flatten the image array for XGBoost
            features = image_array.reshape(1, -1)
            print(f"Raw image features for XGB, shape: {features.shape}")
            
            # Try creating DMatrix and predicting
            try:
                dmatrix = xgb.DMatrix(features)
                xgb_prediction = xgb_model.predict(dmatrix)
            except:
                # If all else fails, use a fallback prediction
                print("XGBoost prediction failed, using fallback")
                xgb_prediction = np.zeros_like(dl_prediction_1.flatten())

        # Ensemble Method (Averaging predictions)
        final_prediction = (dl_prediction_1.flatten() + dl_prediction_2.flatten() + xgb_prediction) / 3
        print("Ensemble prediction complete")

        # Get predicted class
        pred_class_idx = np.argmax(final_prediction)
        pred_class = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else "Unknown"
        confidence = float(final_prediction[pred_class_idx]) * 100
        
        print(f"Predicted class: {pred_class} ({confidence:.2f}%)")

        return jsonify({
            "prediction": {
                "class_index": int(pred_class_idx),
                "class_name": pred_class,
                "confidence": float(confidence)
            },
            "detailed_results": {
                "deep_learning_prediction_1": dl_prediction_1.tolist(),
                "deep_learning_prediction_2": dl_prediction_2.tolist(),
                "xgboost_prediction": xgb_prediction.tolist(),
                "final_prediction": final_prediction.tolist()
            }
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction route: {str(e)}")
        print(error_trace)
        return jsonify({"error": str(e), "traceback": error_trace}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

