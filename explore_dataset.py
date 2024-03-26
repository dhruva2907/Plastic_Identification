from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

def load_trained_model(model_json_path, model_weights_path):
    # Load the trained model architecture from the JSON file
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()

    # Load the model
    loaded_model = model_from_json(model_json)

    # Load the learned weights
    loaded_model.load_weights(model_weights_path)

    return loaded_model

def preprocess_new_image(image_path, target_size):
    # Load and resize the image
    img = Image.open(image_path).resize(target_size)

    # Convert to black and white
    img_bw = img.convert('L')

    # Convert to NumPy array
    img_array = np.array(img_bw)

    # Normalize pixel values
    img_scaled = (img_array / 255.0).astype(np.float32)

    # Expand dimensions to match model input shape
    img_expanded = np.expand_dims(img_scaled, axis=0)

    return img_expanded

def make_predictions_on_image(model, new_data, threshold=0.5):
    # Make predictions 
    predictions = model.predict(new_data)

    # Assuming binary classification, threshold the predictions
    predictions_binary = (predictions > threshold).astype(int)

    return predictions_binary

# Path to the new image
model_json_path = "/home/dhruva/Desktop/VIT/Capstone/model.json"
model_weights_path = "/home/dhruva/Desktop/VIT/Capstone/model_weights.h5"
new_image_path = "/home/dhruva/Desktop/VIT/Capstone/Datasets/try/WhatsApp Image 2023-11-23 at 5.14.21 AM.jpeg"


# Load the trained model
loaded_model = load_trained_model(model_json_path, model_weights_path)

# Preprocess the new image
new_data = preprocess_new_image(new_image_path, target_size=(100, 100))

# Make predictions
predictions = make_predictions_on_image(loaded_model, new_data)

# Visualize the results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(np.array(Image.open(new_image_path)), cmap="gray")  # Use cmap="gray" for black and white images

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(predictions[0, :, :, 0], cmap="gray")

plt.show()
