from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Set up the data generator
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\shash\OneDrive\Desktop\Programming\DeepLearning\data\train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    r'C:\Users\shash\OneDrive\Desktop\Programming\DeepLearning\data\validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the trained model
model.save(r'C:\Users\shash\OneDrive\Desktop\Programming\DeepLearning\new_model')

# Load the trained model for testing on new images
loaded_model = tf.keras.models.load_model(r'C:\Users\shash\OneDrive\Desktop\Programming\DeepLearning\new_model')

# Test the model on a new image
test_image = tf.keras.preprocessing.image.load_img(r'C:\Users\shash\OneDrive\Desktop\Programming\DeepLearning\124.jpg', target_size=(128, 128))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = tf.expand_dims(test_image, axis=0)
prediction = loaded_model.predict(test_image)

# Print the predicted class
if prediction[0][0] >= 0.5:
    print("Object detected")
else:
    print("Object not detected")
