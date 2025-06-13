from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('spinach_leaf_model.h5')  # Or use 'your_model.keras' if saved in Keras format
from tensorflow.keras.models import load_model

# Load the model
model = load_model('spinach_leaf_model.h5')  # Or 'your_model.keras' if saved in Keras format

# Recompile the model to restore metrics
model.compile(
    optimizer='adam',  # Use the same optimizer as during training
    loss='categorical_crossentropy',  # Match the original loss function
    metrics=['accuracy']
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your test dataset
test_dir = "specify your path"

# ImageDataGenerator for test images (No augmentation, just rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test images
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Match input size of MobileNetV2
    batch_size=32,
    class_mode='categorical',  # Change to 'binary' if only two classes
    shuffle=False
)
# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
import numpy as np
from tensorflow.keras.preprocessing import image

# Set the path to the image you want to test
img_path = "your_image_path_here.jpg"  # Replace with your actual image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize like training data

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)  # Get class with highest probability

# Print result
print(f"Predicted Class: {predicted_class}")
print(f"Confidence Scores: {predictions}")
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('spinach_leaf_model.h5')  # Update this with your actual model file path

# Define class labels
class_labels = {
    0: "Anthracnose",
    1: "Bacterial Spot",
    2: "Downy Mildew",
    3: "Healthy Leaf",
    4: "Pest Damage"
}

# Set the test image folder path
test_folder = "path_to_your_test_images_folder"  # Replace with the path to your test folder

# Function to process and predict an image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the class with highest probability
    confidence = predictions[0][predicted_class]  # Confidence score

    return predicted_class, confidence

# Iterate over all images in the folder
for root, _, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
            img_path = os.path.join(root, file)
            predicted_class, confidence = predict_image(img_path)
            print(f"Image: {file} | Predicted Class: {class_labels[predicted_class]} | Confidence: {confidence:.2f}")
