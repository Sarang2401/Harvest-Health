# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np

# Define dataset directory and constants
dataset_dir = ""  # Add the path to your dataset directory
IMAGE_SIZE = 256  # Dimension of images (height and width)
BATCH_SIZE = 32  # Number of samples per batch
CHANNELS = 3  # Number of color channels in images (RGB)
EPOCHS = 50  # Total number of training iterations

# Load dataset and preprocess images into batches
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",  # Path to dataset directory
    shuffle=True,  # Shuffle data before splitting
    image_size=(IMAGE_SIZE, IMAGE_SIZE),  # Resize images to uniform size
    batch_size=BATCH_SIZE  # Process images in batches
)

# Define directories for train, validation, and test sets
train_dir = os.path.join(dataset_dir, "train")  # Training set directory
validation_dir = os.path.join(dataset_dir, "validation")  # Validation set directory
test_dir = os.path.join(dataset_dir, "test")  # Test set directory

# Set parameters for data generators
img_size = (256, 256)  # Uniform image size
BATCH_SIZE = 32  # Batch size
CHANNELS = 3  # Number of color channels
EPOCHS = 50  # Number of epochs

# Create data generators with augmentation (for training)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    rotation_range=20,  # Rotate images randomly
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Zoom into images randomly
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'  # Fill in pixels after transformation
)

# Create data generators for validation and test sets (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Function to partition dataset into train, validation, and test sets
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1,
                              shuffle=True, shuffle_size=10000):
    ds_size = len(ds)  # Get total number of samples
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)  # Shuffle dataset
    train_size = int(train_split * ds_size)  # Calculate train size
    val_size = int(val_split * ds_size)  # Calculate validation size
    train_ds = ds.take(train_size)  # Create training dataset
    val_ds = ds.skip(train_size).take(val_size)  # Create validation dataset
    test_ds = ds.skip(train_size).skip(val_size)  # Create test dataset

    return train_ds, val_ds, test_ds

# Partition dataset
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# Optimize dataset for performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define CNN model for image classification
from tensorflow.keras import layers
n_classes = 3  # Number of classes in dataset
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 256, 3)),  # Input layer with image shape
    tf.keras.layers.Rescaling(1./255),  # Normalize input data
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),  # Convolutional layer
    layers.MaxPooling2D((2, 2)),  # Pooling layer to reduce dimensionality
    layers.Dropout(0.2),  # Dropout for regularization
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Second convolutional layer
    layers.MaxPooling2D((2, 2)),  # Pooling layer
    layers.Dropout(0.2),  # Dropout for regularization
    layers.Flatten(),  # Flatten the feature maps into a 1D vector
    layers.Dense(128, activation='relu'),  # Fully connected dense layer
    layers.Dropout(0.3),  # Dropout for regularization
    layers.Dense(n_classes, activation='softmax'),  # Output layer for classification
])

# Display model summary
model.summary()

# Compile the model with optimizer, loss function, and evaluation metrics
model.compile(
    optimizer='adam',  # Optimizer for model training
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Loss function
    metrics=['accuracy']  # Evaluation metrics
)

# Train the model on the training set and validate on the validation set
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50,  # Number of iterations over the dataset
)

# Evaluate model performance on the test set
scores = model.evaluate(test_ds)

# Plot training and validation accuracy and loss
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

# Display prediction results with images
import numpy as np
class_names = dataset.class_names
for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict:")
    plt.imshow(first_image)
    print("Actual label:", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("Predicted label:", class_names[np.argmax(batch_prediction[0])])

# Function to predict image class and confidence level
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())  # Convert image to array
    img_array = tf.expand_dims(img_array, 0)  # Expand dimensions for model input
    predictions = model.predict(img_array)  # Get predictions from model
    predicted_class = class_names[np.argmax(predictions[0])]  # Determine class
    confidence = round(100 * (np.max(predictions[0])), 2)  # Calculate confidence
    return predicted_class, confidence

# Display predictions for multiple images
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Show the image

        predicted_class, confidence = predict(model, images[i].numpy())  # Predict class and confidence
        actual_class = class_names[labels[i]]  # Get actual class

        plt.title(f"Actual: {actual_class},\nPredicted: {predicted_class}.\nConfidence: {confidence}%")
        plt.axis("off")  # Hide axes

# Save the trained model to disk
model.save("model/plant_model.h5")  # Save model in H5 format
