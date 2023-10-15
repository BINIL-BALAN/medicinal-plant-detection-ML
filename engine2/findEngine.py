
image_array = ["image1","image"]
def recoganice_image(inputImag):
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_hub as hub
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from PIL import Image
    leafName = ""
    custom_objects = {
    'KerasLayer': hub.KerasLayer
    }

    model = tf.keras.models.load_model('D:\project\server\home\engine\my_model.h5', custom_objects=custom_objects)
    print(model)
#     datagen_kwargs = dict(rescale=1./255, validation_split=.80)
#     valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
#     valid_generator = valid_datagen.flow_from_directory(
# '../../../test dataset',
# subset="validation",
# shuffle=True,
# target_size=(224,224)
# )
    # image_path = 'D:\project\server\home\engine\image.jpg'
# print(image_path)
# Load the image and preprocess it
    # image = load_img(image_path, target_size=(224, 224))
    # image = Image.open(image_path)
    image = Image.open(inputImag)
    print(inputImag)
    # image = inputImag
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize the pixel values to be between 0 and 1
    image = image.reshape((1, 224, 224, 3)) 
    predictions = model.predict(image)
    print(predictions)
    predicted_class_index = np.argmax(predictions, axis=1)
    with open('D:\project\server\home\engine\labels.txt', 'r') as f:
        labels = f.read().split('\n')
        predicted_class_name = labels[predicted_class_index[0]]
        leafName = predicted_class_name
        print(predicted_class_name)
    return leafName