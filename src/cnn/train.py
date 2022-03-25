#! /usr/bin/env python
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from math import ceil

# where your source images are stored
IMAGES_DIR = "../resources/images"
# where you have your "training" and "validation" splits
# you can unzip the supplied hotdogs.zip and notdogs.zip into your IMAGES_DIR if you like
# and then use split_data.py util to create splits
TRAINING_DIR = os.path.join(IMAGES_DIR,"training")
VALIDATION_DIR = os.path.join(IMAGES_DIR,"validation")
# hot- and not-dogs
HOTDOG_SOURCE_DIR = os.path.join(IMAGES_DIR,"hotdogs")
HOTDOG_TRAINING_DIR = os.path.join(TRAINING_DIR,"hotdogs")
HOTDOG_VALIDATION_DIR = os.path.join(VALIDATION_DIR,"hotdogs")
NOTDOG_SOURCE_DIR = os.path.join(IMAGES_DIR,"notdogs")
NOTDOG_TRAINING_DIR = os.path.join(TRAINING_DIR,"notdogs")
NOTDOG_VALIDATION_DIR = os.path.join(VALIDATION_DIR,"notdogs")
CHECKPOINT_DIR = '../../checkpoints' 
MODELS_DIR = '../../models' 

NUM_TRAINING_EPOCHS=30
BATCH_SIZE = 20
# desired image height and width
TARGET_SIZE = 224

# computed values
NUM_TRAINING_SAMPLES = len(os.listdir(HOTDOG_TRAINING_DIR)) + len(os.listdir(NOTDOG_TRAINING_DIR))
print("Number of training samples: " + str(NUM_TRAINING_SAMPLES))
STEPS_PER_EPOCH = ceil(NUM_TRAINING_SAMPLES/BATCH_SIZE)
print("Steps per epoch: " + str(STEPS_PER_EPOCH))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(TARGET_SIZE, TARGET_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(2048, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

model.summary()

# scale pixel vals to 0,1 range and randomly adjust each image in each epoch
# should reduce overfitting and mimic online learning
train_datagen = ImageDataGenerator(
    rescale = 1.0/255.,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
 )

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=(TARGET_SIZE, TARGET_SIZE)) 

validation_datagen = ImageDataGenerator( rescale = 1.0/255. )

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                                    target_size=(TARGET_SIZE, TARGET_SIZE)) 

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_DIR,
                                                 save_weights_only=True,
                                                 save_freq=5*BATCH_SIZE,
                                                 verbose=1)

history = model.fit_generator(train_generator,
                              epochs=NUM_TRAINING_EPOCHS,
                              callbacks=[cp_callback],
                              steps_per_epoch=STEPS_PER_EPOCH,
                              verbose=1,
                              validation_data=validation_generator)

model.save(MODELS_DIR)

# plot training vs. validation accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')

plt.show()
