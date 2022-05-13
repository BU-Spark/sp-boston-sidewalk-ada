import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

TRAIN_CLASS_PATHS = ('batches/concrete/color/train',
        'batches/asphalt/color/train',
        'batches/brick/color/train')

model = models.Sequential()
base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(480, 640, 3))
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(TRAIN_CLASS_PATHS), activation='softmax'))

model.save('model')
