import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import PIL
import PIL.Image
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# specify the file path
file_path = "model.h5"

# check if the file exists
if os.path.isfile(file_path):
    print("Loading model")
    cnn = tf.keras.models.load_model("model.h5")

else:
    print("Creating model")
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=[128,128,3]),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


cnn.fit(training_set, validation_data=test_set, epochs=1)
cnn.save("model.h5")
cnn.save("model2.h5")


predcount = 0
count = 0
for i in range(count):
    img=4201+i
    test_image_cat = PIL.Image.open(f'dataset/test_set/cats/cat.{img}.jpg')
    test_cat = test_image_cat.resize((128,128))
    test_cat = np.array(test_cat)
    test_cat = np.expand_dims(test_cat, axis=0)

    result = cnn.predict(test_cat)
    training_set.class_indices
    if result[0][0] == 0:
        predcount+=1
        
    test_image_dog = PIL.Image.open(f'dataset/test_set/dogs/dog.{img}.jpg')
    test_cat = test_image_cat.resize((128,128))
    test_cat = np.array(test_cat)
    test_cat = np.expand_dims(test_cat, axis=0)

    result = cnn.predict(test_cat)
    training_set.class_indices
    if result[0][0] == 1:
        predcount+=1


print(predcount)
