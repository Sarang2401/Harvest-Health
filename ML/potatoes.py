import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np


# Define dataset directory
dataset_dir = "" #Add Your path
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
dataset = tf.keras.preprocessing.image_dataset_from_directory(
 "PlantVillage" ,
 shuffle=True,
 image_size = (IMAGE_SIZE,IMAGE_SIZE),
 batch_size = BATCH_SIZE
)


# Create directories for train, validation, and test sets
train_dir = os.path.join(dataset_dir, "train")
validation_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")


# Define image size
img_size = (256, 256)
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50


# Create data generators with data augmentation
train_datagen = ImageDataGenerator(
 rescale=1./255,
 rotation_range=20,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1,
shuffle=True,shuffle_size=10000):
 ds_size=len(ds)
 if shuffle:
 ds =ds.shuffle(shuffle_size,seed=12)
 train_size= int(train_split*ds_size)
 val_size =int(val_split*ds_size)
 train_ds = ds.take(train_size)
 val_ds=ds.skip(train_size).take(val_size)
 test_ds=ds.skip(train_size).skip(val_size)

 return train_ds, val_ds, test_ds



train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds= test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


from tensorflow.keras import layers
n_classes = 3
model = tf.keras.Sequential([
 tf.keras.layers.Input(shape=(256, 256, 3)), # Input layer
 tf.keras.layers.Rescaling(1./255),
 # Convolutional layers with experimentation options
 layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'), # Padding for preserving
spatial dimensions
 layers.MaxPooling2D((2, 2)),
 layers.Dropout(0.2),
 layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # Experiment with different filter depths
 layers.MaxPooling2D((2, 2)),
 layers.Dropout(0.2),
 # Add more Conv2D and Pooling layers as needed
 layers.Flatten(), # Flatten the feature maps
 # Dense layers with experimentation options
 layers.Dense(128, activation='relu'), # Experiment with number of units
 layers.Dropout(0.3),
 layers.Dense(n_classes, activation='softmax'), # Output layer for multi-class classification
])


model.summary()

model.compile(
 optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
 metrics=['accuracy']
)
history = model.fit(
 train_ds,
 batch_size=BATCH_SIZE,
 validation_data=val_ds,
 verbose=1,
 epochs=50,
)


scores = model.evaluate(test_ds)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


import numpy as np
class_names = dataset.class_names
class_names
for images_batch, labels_batch in test_ds.take(1):

 first_image = images_batch[0].numpy().astype('uint8')
 first_label = labels_batch[0].numpy()

 print("first image to predict")
 plt.imshow(first_image)
 print("actual label:",class_names[first_label])

 batch_prediction = model.predict(images_batch)
 print("predicted label:",class_names[np.argmax(batch_prediction[0])])
 
 
def predict(model, img):
 img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
 img_array = tf.expand_dims(img_array, 0)
 predictions = model.predict(img_array)
 predicted_class = class_names[np.argmax(predictions[0])]
 confidence = round(100 * (np.max(predictions[0])), 2)
 return predicted_class, confidence
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
 for i in range(9):
 ax = plt.subplot(3, 3, i + 1)
 plt.imshow(images[i].numpy().astype("uint8"))

 predicted_class, confidence = predict(model, images[i].numpy())
 actual_class = class_names[labels[i]]

 plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

 plt.axis("off")
 
 model.save("model/plant_model.h5")
