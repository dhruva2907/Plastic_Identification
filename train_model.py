from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from sklearn.metrics import f1_score


# Define batch size
batch_size = 32 #adjust this based on your preference and available memory


def unet_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', input_shape=(100, 100, 1))(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Decoder
    up3 = UpSampling2D(size=(2, 2))(conv3)
    up3 = Conv2D(256, 2, activation='relu', padding='same')(up3)
    merge3 = Concatenate(axis=3)([conv2, up3])
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up4 = UpSampling2D(size=(2, 2))(conv4)
    up4 = Conv2D(64, 2, activation='relu', padding='same')(up4)
    merge4 = Concatenate(axis=3)([conv1, up4])
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge4)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=output)
    
    return model






# Update the paths
dataset_path = '/home/dhruva/Desktop/VIT/Capstone/Datasets/Dataset/'
output_path = '/home/dhruva/Desktop/VIT/Capstone/Datasets/Dataset/Masks'

# Assuming raw and preprocessed images are in 'Images' folder inside the dataset path
images_folder = 'Images'
all_image_filenames = os.listdir(os.path.join(dataset_path, images_folder))

# Split the filenames into training and validation sets
train_filenames, val_filenames = train_test_split(all_image_filenames, test_size=0.2, random_state=42)

# Print some debug information
print("First 5 Training filenames:", train_filenames[:5])  # Print the first 5 filenames for debugging
print("First 5 Validation filenames:", val_filenames[:5])  # Print the first 5 filenames for debugging

# Print the number of images in each set
print(f"Total images: {len(all_image_filenames)}")
print(f"Training images: {len(train_filenames)}")
print(f"Validation images: {len(val_filenames)}")

# Ensure the output folders exist
os.makedirs(output_path, exist_ok=True)

# Function to preprocess image
def preprocess_image(image_path, target_size):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None  # or handle the missing file case accordingly

    # Open the image
    img = Image.open(image_path)

    # Resize the image (adjust the size as needed)
    img_resized = img.resize(target_size)

    # Convert to black and white
    img_bw = img_resized.convert('L')

    # Convert the image to a NumPy array
    img_array = np.array(img_bw)

    # Ensure pixel values are in the range 0 to 255
    img_scaled = (img_array / 255.0).astype(np.float32)


    # Convert the NumPy array back to a Pillow image
    img_final = Image.fromarray(img_scaled)

    return img_final

    
def custom_data_generator(image_folder, mask_folder, filenames, batch_size, target_size):
    while True:
        for i in range(0, len(filenames), batch_size):
            batch_filenames = filenames[i:i + batch_size]

            images = []
            masks = []

            for filename in batch_filenames:
                # Process raw image
                raw_image_path = os.path.join(image_folder, filename)
                raw_mask_path = os.path.join(mask_folder, 'mask_' + filename)

                if os.path.exists(raw_image_path) and os.path.exists(raw_mask_path):
                    raw_img = preprocess_image(raw_image_path, target_size=target_size)
                    raw_mask = preprocess_image(raw_mask_path, target_size=target_size)

                    if raw_img is not None and raw_mask is not None:
                        images.append(np.array(raw_img))
                        masks.append(np.array(raw_mask))

                # Process preprocessed image
                preprocessed_filename = filename.replace('_preprocessed', '')
                preprocessed_image_path = os.path.join(image_folder, preprocessed_filename)
                preprocessed_mask_path = os.path.join(mask_folder, 'mask_' + preprocessed_filename)

                if os.path.exists(preprocessed_image_path) and os.path.exists(preprocessed_mask_path):
                    preprocessed_img = preprocess_image(preprocessed_image_path, target_size=target_size)
                    preprocessed_mask = preprocess_image(preprocessed_mask_path, target_size=target_size)

                    if preprocessed_img is not None and preprocessed_mask is not None:
                        images.append(np.array(preprocessed_img))
                        masks.append(np.array(preprocessed_mask))

            yield np.array(images), np.array(masks)





# Specify the target size for images
target_size = (100,100)

# Get list of image filenames
all_image_filenames = os.listdir(os.path.join(dataset_path, 'Images'))
image_filenames = [filename for filename in all_image_filenames if filename.endswith(('.png', '.jpg', '.jpeg'))]

# Split the filenames into training and validation sets
train_filenames, val_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)

    # Create a custom data generator for training
train_generator = custom_data_generator(    
        image_folder=os.path.join(dataset_path, 'Images'),
        mask_folder=os.path.join(dataset_path, 'Masks'),
        filenames=train_filenames,  
        batch_size=batch_size,
        target_size=target_size
    )

    # Specify the input shape based on your target size
input_shape = target_size + (1,)

    # Create the U-Net model and get intermediate layers
model, conv1, conv2, conv3, conv4, conv5 = unet_model(input_shape)


    # Get the number of training steps
num_train_steps = len(train_filenames) // batch_size  # adjust accordingly

    # Create a custom data generator for validation
val_generator = custom_data_generator(
        image_folder=os.path.join(dataset_path, 'Images'),
        mask_folder=os.path.join(dataset_path, 'Masks'),
        filenames=val_filenames,
        batch_size=batch_size,  # same batch size as the training generator
        target_size=target_size
    )

    # Get the number of validation steps
num_val_steps = len(val_filenames) // batch_size # adjust accordingly

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Create the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)   

 # Train the model with early stopping
model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=num_train_steps,
    validation_data=val_generator,
    validation_steps=num_val_steps,
    callbacks=[early_stopping]
)

# Evaluate the model on the validation set
evaluation_metrics = model.evaluate(val_generator, steps=num_val_steps)

# Extracting the predictions for each batch in the validation set
y_true = []
y_pred = []

for i in range(num_val_steps):
    batch_data, batch_labels = next(val_generator)
    batch_predictions = model.predict(batch_data)
    
    #adjust accordingly
    y_true.extend(batch_labels.flatten())
    y_pred.extend(batch_predictions.flatten())

# Convert predictions to binary values based on a threshold (e.g., 0.5)
threshold = 0.5
y_pred_binary = (np.array(y_pred) > threshold).astype(int)

# Compute the F1 score
f1 = f1_score(y_true, y_pred_binary)

print("F1 score: (including F1 score):", evaluation_metrics + [f1])

# Save the trained model architecture to a JSON file
model_json_path = "/home/dhruva/Desktop/VIT/Capstone/model.json"
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

# Save the learned weights to an HDF5 file
model_weights_path = "/home/dhruva/Desktop/VIT/Capstone/Datasets/Dataset/model_weights.h5"
model.save_weights(model_weights_path)


new_image_path = '/home/dhruva/Desktop/VIT/Capstone/Datasets/Dataset/'  # input directory
for filename in os.listdir(new_image_path):
    image_path = os.path.join(new_image_path, filename)
    new_data = preprocess_image(image_path, target_size)

    # Reshape the data to match the input shape of the model
    new_data = np.expand_dims(new_data, axis=0)

    predictions = model.predict(new_data)
    print(f"Predictions for {filename}:", predictions)



