import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Augmentation parameters (optional)
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'D:\Project\project-server\Arali',
    target_size=(224, 224),  # Resize images to a common size
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)
num_classes = 80
model = models.Sequential([
    keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),  # Adjust input shape based on your dataset
        include_top=False,  # Exclude the top (classification) layer
        weights='imagenet'  # Use pre-trained weights (optional)
    ),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs
    validation_data=validation_generator  # If you have a validation set
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

model.save('custom_model.h5')  # Save the model
loaded_model = keras.models.load_model('custom_model.h5')  # Load the model
