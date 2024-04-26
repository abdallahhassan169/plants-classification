from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
loaded_model = tf.keras.models.load_model('model.h5')

# Define route to receive image and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
    # Receive image file
        file = request.files['image']
        print("reached")
        # Read image
        img = Image.open(file.stream)
        
        # Preprocess image
        img = img.resize((224, 224))  # Resize image to match model's expected sizing
        img = np.array(img) / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make predictions
        predictions = loaded_model.predict(img)
        
        # Decode predictions (assuming binary classification)
        class_names = ['Aglaonema Plant', 'Spider Plant']  # Replace with your class names
        predicted_class = class_names[int(np.round(predictions[0]))]
        info = [  """Aglaonema, also known as Chinese evergreen, is a genus of flowering plants native to tropical and subtropical regions of Asia. They are prized as ornamental houseplants for their attractive foliage and adaptability to indoor conditions. With various cultivars available, Aglaonema plants exhibit a range of leaf shapes, colors, and patterns, making them versatile and popular choices for indoor gardens.""" , "The spider plant (Chlorophytum comosum) is a popular indoor plant with long, arching leaves and small plantlets that dangle from stems, prized for its easy care and air-purifying qualities." ]
        full = {"Aglaonema" : str(1-predictions[0])[1:7] , "Spider": str(predictions[0])[1:7] }
        print(predicted_class)
        # Return predictions
        return jsonify({'prediction': predicted_class ,'full-prediction':full , 'more':info[int(np.round(predictions[0]))]})
    except Exception  as e:
        return jsonify({"err":e})
    

@app.route('/a', methods=['get'])
def pred():
    return "aaaaa"
# Run the app
if __name__ == '__main__':
    app.run(debug=True)