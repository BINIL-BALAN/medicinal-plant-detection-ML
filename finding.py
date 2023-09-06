import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model('./models/my_model.h5')

# Load and preprocess the input image

img_path = './test5.jpg'  # Replace with the path to your input image
print(img_path)
img = image.load_img(img_path, target_size=(224, 224))  # Resize to the same size used during training
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.0  # Normalize pixel values to [0, 1]

predictions = model.predict(img)
predicted_class = np.argmax(predictions, axis=1)[0]
print(predicted_class)

class_labels = {
    0: 'class_0',
    1: 'class_1',
    2: 'class_2',
    3: 'class_3',
    49: 'class_4',
    # Add mappings for all your classes
}

# predicted_label = class_labels[predicted_class]
# print(f"Predicted class: {predicted_label}")