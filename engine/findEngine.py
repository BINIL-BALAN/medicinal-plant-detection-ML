import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
image_array = ["image1","image"]
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = tf.keras.models.load_model('D:\project\server\medicinal-plant-detection-ML\engine\my_model.h5', custom_objects=custom_objects)
print(model)

# datagen_kwargs = dict(rescale=1./255, validation_split=.80)
# valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
# valid_generator = valid_datagen.flow_from_directory(
# 'D:\project\test dataset',
# subset="validation",
# shuffle=True,
# target_size=(224,224)
# )
index = int(input("ennter index"))

image_path = image_array[index] + ".jpg"
print(image_path)
# Load the image and preprocess it
# image = load_img(image_path, target_size=(224, 224))
image = Image.open(image_path)
image = image.resize((224, 224))
image = img_to_array(image)
image = image / 255.0  # Normalize the pixel values to be between 0 and 1
image = image.reshape((1, 224, 224, 3)) 

predictions = model.predict(image)
print(predictions)

predicted_class_index = np.argmax(predictions, axis=1)
with open('D:\project\server\medicinal-plant-detection-ML\labels.txt1', 'r') as f:
    labels = f.read().split('\n')
predicted_class_name = labels[predicted_class_index[0]]
print(predicted_class_name)